"""Modelo unificado KIE (1 backbone): SER + RE treinados conjuntamente.

Diferença de `LiLTRobertaLikeForJointKIE`:
- Este tem **1 único backbone** (self.lilt) compartilhado entre SER e RE
- Requer treino joint (multi-task) para que o backbone aprenda features úteis
  para as duas tarefas simultaneamente — não pode ser construído via fusão
  pós-hoc dos modelos SER e RE separados (backbones divergiriam pós fine-tune)

Fluxo de treino (joint):
1. backbone processa input_ids + bbox → hidden_states
2. classifier → ser_logits (loss_ser via labels gold)
3. extractor recebe concat(hidden, layout) + entities gold + relations gold
   → loss_re via REDecoder
4. loss total = α·loss_ser + β·loss_re

Fluxo de inferência end-to-end:
1. backbone → hidden_states
2. classifier → ser_logits → agregação Python (majority vote) → entities Q/A
3. extractor → pares Q→A classificados

Vantagens sobre `LiLTRobertaLikeForJointKIE` (2 backbones):
- ~50% menos params (287M vs 571M)
- ~50% menos VRAM
- Latência ~30-40% menor (1 forward do backbone)
- ONNX exportável em 2 arquivos (backbone+classifier, re_core)

Desvantagem:
- Requer retraining joint (não pode usar checkpoints SER/RE separados como estão)
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

try:
    from transformers.file_utils import ModelOutput
except ImportError:
    from transformers.utils import ModelOutput

from ...modules.decoders.re import REDecoder
from .modeling_LiLTRobertaLike import (
    LiLTRobertaLikeModel,
    LiLTRobertaLikePreTrainedModel,
)


@dataclass
class JointUnifiedOutput(ModelOutput):
    """Saída do modelo joint unificado (1 backbone).

    - loss: loss combinada (α·loss_ser + β·loss_re) quando labels fornecidos.
    - loss_ser: componente SER da loss (None em inferência).
    - loss_re: componente RE da loss (None em inferência).
    - ser_logits: logits token-level do SER (B, T, num_labels).
    - pred_relations: lista (len=batch) de listas de pares preditos pelo REDecoder.
    - entities_used: entidades efetivamente usadas no RE.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_ser: Optional[torch.FloatTensor] = None
    loss_re: Optional[torch.FloatTensor] = None
    ser_logits: Optional[torch.FloatTensor] = None
    pred_relations: Optional[List[List[dict]]] = None
    entities_used: Optional[List[dict]] = None


class LiLTRobertaLikeForJoint(LiLTRobertaLikePreTrainedModel):
    """Modelo joint unificado SER+RE com 1 backbone compartilhado.

    Pesos esperados no state_dict:
    - `lilt.*` — backbone único (treinado em joint, não é cópia do SER)
    - `classifier.*` — SER head (Linear 768→num_labels)
    - `extractor.*` — RE head (REDecoder com use_vectorized=True)

    Parâmetros:
    - config.num_labels: quantas classes SER (tipicamente 5 para BIO O/B-Q/B-A/I-A/I-Q)
    - ser_loss_weight (α): peso da loss SER na combinação (default 1.0)
    - re_loss_weight (β): peso da loss RE na combinação (default 1.0)
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # SER label scheme (igual HF LiltForTokenClassification e LiLTRobertaLikeForJointKIE)
    _SER_LABEL_O = 0
    _SER_LABEL_Q_IDS = (1, 4)  # B-QUESTION, I-QUESTION
    _SER_LABEL_A_IDS = (2, 3)  # B-ANSWER, I-ANSWER

    def __init__(self, config, ser_loss_weight: float = 1.0, re_loss_weight: float = 1.0):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 5)
        self.ser_loss_weight = ser_loss_weight
        self.re_loss_weight = re_loss_weight

        # Backbone único (compartilhado SER + RE)
        self.lilt = LiLTRobertaLikeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        hidden_concat_size = (
            config.hidden_size + config.hidden_size // config.channel_shrink_ratio
        )

        # SER head — Linear direto no sequence_output (768 dims)
        # Segue o HF nativo LiltForTokenClassification (não o jpWang que concatena layout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # RE head — REDecoder vetorizado (ONNX-friendly)
        # Recebe concat (sequence_output, layout_outputs) = 960 dims
        self.extractor = REDecoder(config, hidden_concat_size, use_vectorized=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # KIE-specific args
        labels=None,              # gold SER labels (B, T) — para treino
        entities=None,            # gold entities (list of dict {start, end, label}) — para treino e inferência com entities conhecidas
        relations=None,           # gold relations (list of dict {head, tail}) — para treino
        token_span_map=None,      # para inferência automática (deriva entities do SER)
    ):
        """Forward joint.

        MODO TREINO (labels + entities + relations fornecidos):
          - Retorna loss combinada (α·loss_ser + β·loss_re)
          - ser_logits são calculados mas só contribuem via loss_ser
          - entities/relations são gold (do data_collator), não derivadas do SER

        MODO INFERÊNCIA (só input_ids + bbox + attention_mask + token_span_map):
          - Agrega ser_logits → entities Q/A via majority vote
          - Roda extractor sobre pares Q×A candidatos
          - Retorna pred_relations

        MODO INFERÊNCIA COM ENTITIES CONHECIDAS (fornece entities, omite labels/relations):
          - Pula agregação SER
          - Usa entities passadas para o extractor
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, layout_outputs = self.lilt(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]                                          # (B, T, 768)
        hidden_concat = torch.cat([sequence_output, layout_outputs], -1)     # (B, T, 960)

        # SER head — usa só sequence_output (768 dims)
        ser_logits = self.classifier(self.dropout(sequence_output))

        # Loss SER (só em treino)
        loss_ser = None
        if labels is not None:
            loss_fct_ser = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ser_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct_ser.ignore_index).type_as(labels),
                )
                loss_ser = loss_fct_ser(active_logits, active_labels)
            else:
                loss_ser = loss_fct_ser(ser_logits.view(-1, self.num_labels), labels.view(-1))

        # Modo inferência automática: derivar entities via agregação SER
        if entities is None:
            if token_span_map is None:
                raise ValueError(
                    "Quando entities não é fornecido, é obrigatório passar "
                    "token_span_map (List[List[int]] mapeando cada token para span_idx, "
                    "ou -1 para tokens não-span). No treino, entities vem do data_collator."
                )
            entities = self._aggregate_ser_to_entities(ser_logits, token_span_map)

        # Em inferência sem gold relations, passar lista vazia para o REDecoder popular pares
        if relations is None:
            relations = [{"head": [], "tail": []} for _ in range(len(entities))]

        # RE head — REDecoder vetorizado
        re_input = self.dropout(hidden_concat)
        loss_re, pred_relations = self.extractor(re_input, entities, relations)

        # Loss combinada
        total_loss = None
        if loss_ser is not None or (loss_re is not None and loss_re != 0):
            total_loss = 0.0
            if loss_ser is not None:
                total_loss = total_loss + self.ser_loss_weight * loss_ser
            if loss_re is not None:
                total_loss = total_loss + self.re_loss_weight * loss_re

        return JointUnifiedOutput(
            loss=total_loss,
            loss_ser=loss_ser,
            loss_re=loss_re if (loss_re is not None and not isinstance(loss_re, int)) else None,
            ser_logits=ser_logits,
            pred_relations=pred_relations,
            entities_used=entities,
        )

    def _aggregate_ser_to_entities(self, ser_logits, token_span_map):
        """Agrega predições SER token-level → entities Q/A por span (majority vote).

        Idêntica à implementação em LiLTRobertaLikeForJointKIE.
        """
        preds = ser_logits.argmax(-1).cpu().tolist()
        results = []
        for preds_b, t2s_b in zip(preds, token_span_map):
            span_tokens = defaultdict(list)
            span_positions = defaultdict(list)
            for tok_idx, (p, s) in enumerate(zip(preds_b, t2s_b)):
                if s is not None and s >= 0:
                    span_tokens[s].append(p)
                    span_positions[s].append(tok_idx)

            entity_starts, entity_ends, entity_labels = [], [], []
            for span_idx in sorted(span_tokens.keys()):
                counter = Counter()
                for t in span_tokens[span_idx]:
                    if t == self._SER_LABEL_O:
                        counter["O"] += 1
                    elif t in self._SER_LABEL_Q_IDS:
                        counter["Q"] += 1
                    elif t in self._SER_LABEL_A_IDS:
                        counter["A"] += 1
                if not counter:
                    continue
                label = counter.most_common(1)[0][0]
                if label == "O":
                    continue
                positions = span_positions[span_idx]
                entity_starts.append(positions[0])
                entity_ends.append(positions[-1] + 1)
                entity_labels.append(0 if label == "Q" else 1)

            if len(entity_starts) < 2:
                entity_starts = [0, 1]
                entity_ends = [1, 2]
                entity_labels = [0, 1]

            results.append({
                "start": entity_starts,
                "end": entity_ends,
                "label": entity_labels,
            })
        return results
