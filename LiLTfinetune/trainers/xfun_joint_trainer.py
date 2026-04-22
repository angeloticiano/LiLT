"""Trainer para o modelo joint unified (LiLTRobertaLikeForJoint).

Estende XfunReTrainer para:
- Aceitar `labels` (SER gold) + `entities` + `relations` no batch
- Coletar ser_logits e pred_relations no prediction_loop
- Computar métricas: eval_f1_ser (seqeval BIO), eval_f1_re (re_score), eval_combined_f1 (média)
- Loggar loss_ser e loss_re separados além de loss total

metric_for_best_model deve ser 'eval_combined_f1' para load_best_model_at_end
pegar o checkpoint com melhor equilíbrio entre as duas tarefas.
"""
from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.utils import logging
from transformers.trainer_utils import EvalPrediction, PredictionOutput

from .xfun_trainer import XfunReTrainer
from ..evaluation import re_score

logger = logging.get_logger(__name__)


class XfunJointTrainer(XfunReTrainer):
    """Trainer para treino conjunto SER+RE no modelo LiLTRobertaLikeForJoint.

    Args específicos via trainer args:
    - metric_for_best_model = "eval_combined_f1"
    - label_names passados: ["labels", "relations"] (automático via super().__init__)

    Args do modelo:
    - ser_loss_weight, re_loss_weight já configurados no modelo (não no trainer)
    """

    SER_BIO_LABELS = ["O", "B-QUESTION", "B-ANSWER", "I-ANSWER", "I-QUESTION"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Garantir que labels e relations apareçam em label_names
        for name in ("labels", "relations"):
            if name not in self.label_names:
                self.label_names.append(name)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Passa entities/relations/labels para o modelo."""
        inputs = self._prepare_inputs(inputs)

        # Garantir que entities e relations estejam no input
        # (o _prepare_inputs move tensors para device mas mantém listas Python)

        with torch.no_grad():
            _use_amp = getattr(self, "use_amp", False) or getattr(self, "use_apex", False)
            if _use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Retornar: (outputs, (ser_labels, relations))
        # O prediction_loop vai separar depois
        ser_labels = inputs.get("labels")
        re_labels = inputs.get("relations")
        return outputs, (ser_labels, re_labels)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Prediction loop customizado: coleta ser_logits + pred_relations juntos.

        Computa:
        - eval_f1_ser: F1 BIO via seqeval (mesma métrica de LiLTRobertaLikeForTokenClassification)
        - eval_f1_re: F1 via re_score (mesma do XfunReTrainer)
        - eval_combined_f1: média das duas (critério para best checkpoint)
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        model = self._wrap_model(self.model, training=False)
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s (joint) *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()
        self.callback_handler.eval_dataloader = dataloader

        # Acumuladores
        all_ser_logits = []
        all_ser_labels = []
        all_re_labels = None
        all_pred_relations = None
        all_entities = None
        total_loss = 0.0
        total_loss_ser = 0.0
        total_loss_re = 0.0
        n_steps = 0

        for step, inputs in enumerate(dataloader):
            outputs, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            ser_labels, re_labels = labels

            # SER
            if outputs.ser_logits is not None and ser_labels is not None:
                all_ser_logits.append(outputs.ser_logits.detach().cpu())
                all_ser_labels.append(ser_labels.detach().cpu())

            # RE (acumular igual ao XfunReTrainer)
            if all_re_labels is None:
                all_re_labels = re_labels
                all_pred_relations = outputs.pred_relations
                all_entities = outputs.entities_used
            else:
                all_re_labels = all_re_labels + re_labels
                all_pred_relations = all_pred_relations + outputs.pred_relations
                all_entities = all_entities + outputs.entities_used

            # Losses
            if outputs.loss is not None:
                total_loss += float(outputs.loss.detach().cpu().item())
            if outputs.loss_ser is not None:
                total_loss_ser += float(outputs.loss_ser.detach().cpu().item())
            if outputs.loss_re is not None:
                total_loss_re += float(outputs.loss_re.detach().cpu().item())
            n_steps += 1

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        # --- Métrica SER (BIO via seqeval) ---
        ser_metrics = self._compute_ser_metrics(all_ser_logits, all_ser_labels)

        # --- Métrica RE (re_score, mesma do XfunReTrainer) ---
        gt_relations = []
        for b in range(len(all_re_labels)):
            rel_sent = []
            ent_b = all_entities[b]
            for head, tail in zip(all_re_labels[b]["head"], all_re_labels[b]["tail"]):
                rel = {
                    "head_id": head,
                    "head": (ent_b["start"][head], ent_b["end"][head]),
                    "head_type": ent_b["label"][head],
                    "tail_id": tail,
                    "tail": (ent_b["start"][tail], ent_b["end"][tail]),
                    "tail_type": ent_b["label"][tail],
                    "type": 1,
                }
                rel_sent.append(rel)
            gt_relations.append(rel_sent)

        re_scores = re_score(all_pred_relations, gt_relations, mode="boundaries")
        re_metrics = {
            "f1": re_scores["ALL"]["f1"],
            "precision": re_scores["ALL"]["p"],
            "recall": re_scores["ALL"]["r"],
        }

        # --- Combinar ---
        combined_f1 = 0.5 * ser_metrics["f1"] + 0.5 * re_metrics["f1"]

        metrics = {
            f"{metric_key_prefix}_f1_ser": ser_metrics["f1"],
            f"{metric_key_prefix}_precision_ser": ser_metrics["precision"],
            f"{metric_key_prefix}_recall_ser": ser_metrics["recall"],
            f"{metric_key_prefix}_f1_re": re_metrics["f1"],
            f"{metric_key_prefix}_precision_re": re_metrics["precision"],
            f"{metric_key_prefix}_recall_re": re_metrics["recall"],
            f"{metric_key_prefix}_combined_f1": combined_f1,
            f"{metric_key_prefix}_loss": total_loss / max(n_steps, 1),
            f"{metric_key_prefix}_loss_ser": total_loss_ser / max(n_steps, 1),
            f"{metric_key_prefix}_loss_re": total_loss_re / max(n_steps, 1),
        }
        return metrics

    def _compute_ser_metrics(self, all_logits, all_labels):
        """Calcula F1/P/R SER via seqeval (BIO mode)."""
        from seqeval.metrics import f1_score, precision_score, recall_score

        if not all_logits:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        # Com padding dinâmico (group_by_length), cada batch tem seq_len distinto.
        # Iterar batch a batch em vez de concatenar.
        id_to_label = {i: l for i, l in enumerate(self.SER_BIO_LABELS)}
        all_true = []
        all_pred = []
        for logits_b, labels_b in zip(all_logits, all_labels):
            preds_b = logits_b.argmax(-1).numpy()
            labels_np = labels_b.numpy()
            for i in range(preds_b.shape[0]):
                seq_true = [id_to_label[int(l)] for l in labels_np[i] if l != -100]
                seq_pred = [id_to_label[int(p)] for p, l in zip(preds_b[i], labels_np[i]) if l != -100]
                if seq_true:
                    all_true.append(seq_true)
                    all_pred.append(seq_pred)

        if not all_true:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        return {
            "f1": float(f1_score(all_true, all_pred, zero_division=0)),
            "precision": float(precision_score(all_true, all_pred, zero_division=0)),
            "recall": float(recall_score(all_true, all_pred, zero_division=0)),
        }

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate customizado: roda prediction_loop e loga métricas separadas."""
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.args.local_rank = torch.distributed.get_rank()
        else:
            self.args.local_rank = -1

        start_time = time.time()

        metrics = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        metrics["eval_runtime"] = round(time.time() - start_time, 4)

        # Loggar
        logger.info(
            "  eval_f1_ser=%.4f  eval_f1_re=%.4f  eval_combined_f1=%.4f  "
            "eval_loss=%.4f (ser=%.4f, re=%.4f)",
            metrics["eval_f1_ser"], metrics["eval_f1_re"], metrics["eval_combined_f1"],
            metrics["eval_loss"], metrics["eval_loss_ser"], metrics["eval_loss_re"],
        )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
