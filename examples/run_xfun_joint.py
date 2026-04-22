#!/usr/bin/env python
# coding=utf-8
"""Script de treino joint SER+RE unificado (LiLTRobertaLikeForJoint).

Adaptação do run_xfun_re.py para:
- Carregar LiLTRobertaLikeForJoint (1 backbone compartilhado)
- Usar XfunJointTrainer (compute_metrics SER + RE + combined_f1)
- Aceitar args --ser_loss_weight e --re_loss_weight

Uso típico:
  python jpwang-lilt/examples/run_xfun_joint.py \
    --model_name_or_path SCUT-DLVCLab/lilt-infoxlm-base \
    --tokenizer_name xlm-roberta-base \
    --output_dir output/kie_production_unified_run \
    --do_train --do_eval \
    --lang synth_prod_20k \
    --max_steps 20000 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 --warmup_ratio 0.06 \
    --logging_steps 100 --evaluation_strategy steps \
    --eval_steps 2000 --save_steps 2000 --save_total_limit 3 \
    --metric_for_best_model eval_combined_f1 --load_best_model_at_end \
    --fp16 \
    --ser_loss_weight 1.0 --re_loss_weight 1.0
"""
with open('tag.txt', 'w') as tagf:
    tagf.write('multilingual')

import json as _json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch as _torch
from datasets import ClassLabel, load_dataset

import LiLTfinetune.data.datasets.xfun
import transformers
from LiLTfinetune.data.data_args import XFUNDataTrainingArguments
from LiLTfinetune.data.data_collator import DataCollatorForKeyValueExtraction
from LiLTfinetune.models.LiLTRobertaLike import (
    LiLTRobertaLikeConfig,
    LiLTRobertaLikeForJoint,
)
from LiLTfinetune.models.model_args import ModelArguments
from LiLTfinetune.trainers.xfun_joint_trainer import XfunJointTrainer
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingArguments:
    """Args específicos do treino joint (não cabem em ModelArguments/TrainingArguments)."""
    ser_loss_weight: float = field(
        default=1.0, metadata={"help": "Peso α da loss SER na combinação (α·L_SER + β·L_RE)."}
    )
    re_loss_weight: float = field(
        default=1.0, metadata={"help": "Peso β da loss RE na combinação."}
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, XFUNDataTrainingArguments, TrainingArguments, JointTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, joint_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, joint_args = parser.parse_args_into_dataclasses()

    # --- Detectar último checkpoint ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # --- Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Joint args: ser_loss_weight={joint_args.ser_loss_weight}, re_loss_weight={joint_args.re_loss_weight}")

    set_seed(training_args.seed)

    # --- Dataset ---
    datasets = load_dataset(
        os.path.abspath(LiLTfinetune.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )

    if training_args.do_train:
        features = datasets["train"].features
    else:
        features = datasets["validation"].features
    label_column_name = "labels"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        return sorted(list(unique_labels))

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
    num_labels = len(label_list)
    logger.info(f"num_labels={num_labels} ({label_list})")

    # --- Carregar modelo ---
    # 3 cenários:
    # 1. HF Hub (SCUT-DLVCLab/lilt-infoxlm-base): config HF, remap 'x' → 'lilt.x', SER+RE aleatórios
    # 2. Checkpoint local LiLTRobertaLike (de run_xfun_re ou run_xfun_joint anterior): load direto
    # 3. Checkpoint local HF Lilt (de run_xfun_ser SER best): remap + init head RE aleatório

    _hf_model_id = model_args.model_name_or_path
    _is_hf_hub = not os.path.isdir(_hf_model_id)

    if _is_hf_hub:
        # Caso 1: HF Hub
        _hf_config = AutoConfig.from_pretrained(
            _hf_model_id, cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config = LiLTRobertaLikeConfig(
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            vocab_size=_hf_config.vocab_size,
            hidden_size=_hf_config.hidden_size,
            num_hidden_layers=_hf_config.num_hidden_layers,
            num_attention_heads=_hf_config.num_attention_heads,
            intermediate_size=_hf_config.intermediate_size,
            hidden_act=_hf_config.hidden_act,
            hidden_dropout_prob=_hf_config.hidden_dropout_prob,
            attention_probs_dropout_prob=_hf_config.attention_probs_dropout_prob,
            max_position_embeddings=_hf_config.max_position_embeddings,
            type_vocab_size=_hf_config.type_vocab_size,
            initializer_range=_hf_config.initializer_range,
            layer_norm_eps=_hf_config.layer_norm_eps,
            pad_token_id=_hf_config.pad_token_id,
            bos_token_id=_hf_config.bos_token_id,
            eos_token_id=_hf_config.eos_token_id,
            channel_shrink_ratio=getattr(_hf_config, "channel_shrink_ratio", 4),
            max_2d_position_embeddings=getattr(_hf_config, "max_2d_position_embeddings", 1024),
        )
        model = LiLTRobertaLikeForJoint(
            config,
            ser_loss_weight=joint_args.ser_loss_weight,
            re_loss_weight=joint_args.re_loss_weight,
        )
        _ckpt_path = hf_hub_download(_hf_model_id, "pytorch_model.bin", cache_dir=model_args.cache_dir)
        _hf_state = _torch.load(_ckpt_path, map_location="cpu")
        _model_state = model.state_dict()
        _new_state = {}
        for k, v in _hf_state.items():
            _new_key = "lilt." + k
            if _new_key in _model_state:
                _new_state[_new_key] = v
        _missing = [k for k in _model_state if k not in _new_state]
        logger.info(f"HF Hub init: {len(_new_state)} keys carregadas, {len(_missing)} aleatórias (classifier + extractor)")
        _model_state.update(_new_state)
        model.load_state_dict(_model_state)
    else:
        # Caso 2 ou 3: checkpoint local
        _local_cfg_path = os.path.join(_hf_model_id, "config.json")
        with open(_local_cfg_path) as _f:
            _local_cfg = _json.load(_f)

        model_type = _local_cfg.get("model_type")
        if model_type in ("liltrobertalike", "liltrobertalikejointkie", "liltrobertalikejoint"):
            # Caso 2: local LiLTRobertaLike
            config = LiLTRobertaLikeConfig(
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                **{k: v for k, v in _local_cfg.items()
                   if k not in ("model_type", "architectures", "transformers_version",
                                "id2label", "label2id", "finetuning_task", "num_labels")},
            )
            model = LiLTRobertaLikeForJoint(
                config,
                ser_loss_weight=joint_args.ser_loss_weight,
                re_loss_weight=joint_args.re_loss_weight,
            )
            _ckpt = os.path.join(_hf_model_id, "pytorch_model.bin")
            _state = _torch.load(_ckpt, map_location="cpu")
            _missing, _unexp = model.load_state_dict(_state, strict=False)
            logger.info(f"Local LiLTRobertaLike carregado: {len(_missing)} missing, {len(_unexp)} unexpected")
        elif model_type == "lilt":
            # Caso 3: checkpoint HF Lilt (SER best)
            config = LiLTRobertaLikeConfig(
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                vocab_size=_local_cfg["vocab_size"],
                hidden_size=_local_cfg["hidden_size"],
                num_hidden_layers=_local_cfg["num_hidden_layers"],
                num_attention_heads=_local_cfg["num_attention_heads"],
                intermediate_size=_local_cfg["intermediate_size"],
                hidden_act=_local_cfg["hidden_act"],
                hidden_dropout_prob=_local_cfg["hidden_dropout_prob"],
                attention_probs_dropout_prob=_local_cfg["attention_probs_dropout_prob"],
                max_position_embeddings=_local_cfg["max_position_embeddings"],
                type_vocab_size=_local_cfg["type_vocab_size"],
                initializer_range=_local_cfg["initializer_range"],
                layer_norm_eps=_local_cfg["layer_norm_eps"],
                pad_token_id=_local_cfg["pad_token_id"],
                bos_token_id=_local_cfg["bos_token_id"],
                eos_token_id=_local_cfg["eos_token_id"],
                channel_shrink_ratio=_local_cfg.get("channel_shrink_ratio", 4),
                max_2d_position_embeddings=_local_cfg.get("max_2d_position_embeddings", 1024),
            )
            model = LiLTRobertaLikeForJoint(
                config,
                ser_loss_weight=joint_args.ser_loss_weight,
                re_loss_weight=joint_args.re_loss_weight,
            )
            _ckpt = os.path.join(_hf_model_id, "pytorch_model.bin")
            _hf_state = _torch.load(_ckpt, map_location="cpu")
            _model_state = model.state_dict()
            _new_state = {}
            for k, v in _hf_state.items():
                # SER HF tem 'lilt.*' + 'classifier.*' — ambos são compatíveis!
                # (nosso classifier é Linear 768→5, igual ao SER HF)
                if k in _model_state:
                    _new_state[k] = v
            _missing = [k for k in _model_state if k not in _new_state]
            logger.info(
                f"HF Lilt (SER best) init: {len(_new_state)} keys carregadas "
                f"(backbone + classifier SER), {len(_missing)} aleatórias (extractor RE)"
            )
            _model_state.update(_new_state)
            model.load_state_dict(_model_state)
        else:
            raise ValueError(f"model_type desconhecido: {model_type}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Requires a fast tokenizer.")

    # --- Dataset ---
    padding = "max_length" if data_args.pad_to_max_length else "longest"

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding, max_length=512,
    )

    # --- Trainer ---
    trainer = XfunJointTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, data_collator=data_collator,
    )

    # Libera VRAM/RAM após eval/save — evita fragmentação do alocador CUDA
    import gc as _gc
    from transformers import TrainerCallback as _TrainerCallback
    class _EmptyCacheCallback(_TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
        def on_save(self, args, state, control, **kwargs):
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
    trainer.add_callback(_EmptyCacheCallback())

    # --- Train ---
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    # --- Eval final ---
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
