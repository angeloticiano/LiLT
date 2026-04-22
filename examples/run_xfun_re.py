#!/usr/bin/env python
# coding=utf-8
with open('tag.txt', 'w') as tagf:
    tagf.write('multilingual')
import logging
import os
import sys

import numpy as np
from datasets import ClassLabel, load_dataset

import LiLTfinetune.data.datasets.xfun
import transformers
from LiLTfinetune import AutoModelForRelationExtraction
from LiLTfinetune.data.data_args import XFUNDataTrainingArguments
from LiLTfinetune.data.data_collator import DataCollatorForKeyValueExtraction
from LiLTfinetune.evaluation import re_score
from LiLTfinetune.models.model_args import ModelArguments
from LiLTfinetune.trainers import XfunReTrainer
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    datasets = load_dataset(
        os.path.abspath(LiLTfinetune.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "input_ids"
    label_column_name = "labels"

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # PATCH: SCUT-DLVCLab/lilt-infoxlm-base usa LiltConfig (HF oficial) mas o código
    # jpWang espera LiLTRobertaLikeConfig. Carregamos os pesos manualmente com remapeamento
    # de keys (HF usa raiz, jpWang usa prefixo 'lilt.').
    from LiLTfinetune.models.LiLTRobertaLike import LiLTRobertaLikeConfig, LiLTRobertaLikeForRelationExtraction
    from huggingface_hub import hf_hub_download
    import torch as _torch

    _hf_model_id = model_args.model_name_or_path
    _is_hf_hub = not os.path.isdir(_hf_model_id)

    if _is_hf_hub:
        # Baixar config do HF para pegar os hiperparâmetros
        _hf_config = AutoConfig.from_pretrained(
            _hf_model_id,
            cache_dir=model_args.cache_dir,
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
            channel_shrink_ratio=getattr(_hf_config, 'channel_shrink_ratio', 4),
            max_2d_position_embeddings=getattr(_hf_config, 'max_2d_position_embeddings', 1024),
        )
        # Instanciar modelo com pesos aleatórios
        model = LiLTRobertaLikeForRelationExtraction(config)
        # Baixar e carregar pesos do HF, remapeando 'raiz.' -> 'lilt.'
        _ckpt_path = hf_hub_download(
            _hf_model_id, 'pytorch_model.bin',
            cache_dir=model_args.cache_dir,
        )
        _hf_state = _torch.load(_ckpt_path, map_location='cpu')
        _model_state = model.state_dict()
        _new_state = {}
        for k, v in _hf_state.items():
            _new_key = 'lilt.' + k
            if _new_key in _model_state:
                _new_state[_new_key] = v
            else:
                logger.warning(f"PATCH: key ignorada (não encontrada no modelo): {k}")
        _missing = [k for k in _model_state if k not in _new_state]
        logger.info(f"PATCH: {len(_new_state)} keys carregadas do HF, {len(_missing)} inicializadas aleatoriamente (head RE)")
        _model_state.update(_new_state)
        model.load_state_dict(_model_state)
    else:
        # Checkpoint local — verificar se é LiLTRobertaLike (salvo pelo training loop do jpWang)
        import json as _json
        _local_config_path = os.path.join(model_args.model_name_or_path, "config.json")
        with open(_local_config_path) as _f:
            _local_cfg_dict = _json.load(_f)
        if _local_cfg_dict.get("model_type") == "liltrobertalike":
            config = LiLTRobertaLikeConfig(
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                **{k: v for k, v in _local_cfg_dict.items()
                   if k not in ("model_type", "architectures", "transformers_version",
                                "id2label", "label2id", "finetuning_task")},
            )
            model = LiLTRobertaLikeForRelationExtraction(config)
            import torch as _torch
            _ckpt = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
            _state = _torch.load(_ckpt, map_location="cpu")
            model.load_state_dict(_state)
            logger.info(f"Checkpoint local LiLTRobertaLike carregado de {_ckpt}")
        elif _local_cfg_dict.get("model_type") == "lilt":
            # PATCH RE: checkpoint local veio do SER treinado com AutoModelForTokenClassification (HF lilt nativo).
            # Transferimos os pesos do backbone para LiLTRobertaLikeForRelationExtraction com remap 'lilt.*' → 'lilt.*'
            # (no HF, o LiltForTokenClassification tem prefixo 'lilt.' no body + 'classifier.' no head).
            # O head RE (relation extraction) é inicializado aleatoriamente.
            config = LiLTRobertaLikeConfig(
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                vocab_size=_local_cfg_dict["vocab_size"],
                hidden_size=_local_cfg_dict["hidden_size"],
                num_hidden_layers=_local_cfg_dict["num_hidden_layers"],
                num_attention_heads=_local_cfg_dict["num_attention_heads"],
                intermediate_size=_local_cfg_dict["intermediate_size"],
                hidden_act=_local_cfg_dict["hidden_act"],
                hidden_dropout_prob=_local_cfg_dict["hidden_dropout_prob"],
                attention_probs_dropout_prob=_local_cfg_dict["attention_probs_dropout_prob"],
                max_position_embeddings=_local_cfg_dict["max_position_embeddings"],
                type_vocab_size=_local_cfg_dict["type_vocab_size"],
                initializer_range=_local_cfg_dict["initializer_range"],
                layer_norm_eps=_local_cfg_dict["layer_norm_eps"],
                pad_token_id=_local_cfg_dict["pad_token_id"],
                bos_token_id=_local_cfg_dict["bos_token_id"],
                eos_token_id=_local_cfg_dict["eos_token_id"],
                channel_shrink_ratio=_local_cfg_dict.get("channel_shrink_ratio", 4),
                max_2d_position_embeddings=_local_cfg_dict.get("max_2d_position_embeddings", 1024),
            )
            model = LiLTRobertaLikeForRelationExtraction(config)
            import torch as _torch
            _ckpt = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
            _hf_state = _torch.load(_ckpt, map_location="cpu")
            _model_state = model.state_dict()
            _new_state = {}
            for k, v in _hf_state.items():
                # HF LiltForTokenClassification salva: 'lilt.<body>' + 'classifier.<head>'
                # jpWang LiLTRobertaLikeForRelationExtraction espera: 'lilt.<body>' + 'extractor.<head RE>'
                # Pulamos 'classifier.*' (head SER não serve para RE)
                if k.startswith("classifier."):
                    continue
                if k in _model_state:
                    _new_state[k] = v
                else:
                    logger.warning(f"PATCH RE: key ignorada (não no modelo): {k}")
            _missing = [k for k in _model_state if k not in _new_state]
            logger.info(f"PATCH RE: {len(_new_state)} keys carregadas do SER HF, {len(_missing)} inicializadas aleatoriamente (head RE + eventuais)")
            _model_state.update(_new_state)
            model.load_state_dict(_model_state)
        else:
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model = AutoModelForRelationExtraction.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    def compute_metrics(p):
        pred_relations, gt_relations = p
        score = re_score(pred_relations, gt_relations, mode="boundaries")
        return score

    # Initialize our Trainer
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
