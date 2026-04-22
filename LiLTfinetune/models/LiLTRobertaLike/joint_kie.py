"""Modelo unificado KIE: backbone + SER classifier + RE extractor em um único forward.

Segue o padrão de `LiLTRobertaLikeForTokenClassification` (SER) e
`LiLTRobertaLikeForRelationExtraction` (RE), mas com backbone compartilhado.

Fluxo de inferência end-to-end:
1. backbone (lilt) processa input_ids + bbox
2. classifier gera logits SER (Q/A/O por token)
3. Agrega token predictions → labels por span (via token_span_map)
4. Constrói entities Q/A a partir das labels
5. extractor (REDecoder) classifica cada par Q×A → pares positivos

Fluxo de fine-tuning (não usado por padrão): aceita entities e relations externas,
assim como LiLTRobertaLikeForRelationExtraction faz.
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
class JointKIEOutput(ModelOutput):
    """Saída do modelo unificado.

    - loss: loss RE quando `relations` (gold) é fornecido (fine-tuning); caso contrário None.
    - ser_logits: logits token-level do SER (B, T, num_labels).
    - pred_relations: lista (len=batch) de listas de pares preditos pelo REDecoder.
    - entities_used: entidades efetivamente usadas no RE (úteis para debug/reconstrução).
    """
    loss: Optional[torch.FloatTensor] = None
    ser_logits: Optional[torch.FloatTensor] = None
    pred_relations: Optional[List[List[dict]]] = None
    entities_used: Optional[List[dict]] = None


class LiLTRobertaLikeForJointKIE(LiLTRobertaLikePreTrainedModel):
    """Modelo joint SER+RE com backbone compartilhado.

    Pesos esperados no state_dict:
    - `lilt.*` — backbone (vem do SER best)
    - `classifier.*` — SER head (vem do SER best)
    - `extractor.*` — RE head (vem do RE best)
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # SER label scheme (igual ao usado em LiLTRobertaLikeForTokenClassification)
    # LABEL_LIST = ["O", "B-QUESTION", "B-ANSWER", "I-ANSWER", "I-QUESTION"]
    _SER_LABEL_O = 0
    _SER_LABEL_Q_IDS = (1, 4)  # B-QUESTION, I-QUESTION
    _SER_LABEL_A_IDS = (2, 3)  # B-ANSWER, I-ANSWER

    def __init__(self, config, use_re_vectorized: bool = False):
        """Construtor do modelo joint.

        - use_re_vectorized=False (default): REDecoder usa forward legacy (for-loop Python).
          Compatível com treino original e paridade bit-exata verificada.
        - use_re_vectorized=True: REDecoder usa forward_vectorized (ONNX-friendly).
          Mesmos pesos, mesmos resultados (paridade bit-exata validada em val).
          Use para export ONNX ou deploy com onnxruntime.
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 5)
        self._use_re_vectorized = use_re_vectorized

        # DECISÃO ARQUITETURAL IMPORTANTE (2026-04-21):
        # SER e RE foram treinados em sequência (SER 40k → RE 30k). O fine-tune do RE
        # modificou ~398 dos 399 pesos do backbone (diffs de até 0.03). Ou seja, os
        # dois não compartilham o mesmo backbone na prática.
        #
        # Se usarmos só 1 backbone unificado:
        # - Backbone do SER → classifier funciona, mas extractor RE degrada muito
        #   (medido: 30→2 pares, perda >90% de recall)
        # - Backbone do RE → classifier do SER degrada (labels erradas)
        #
        # Solução: manter 2 backbones (um para SER, um para RE), mas num único modelo.
        # Custo extra: ~1GB no disco, ~1GB VRAM. Benefício: 1 load, 1 forward unificado,
        # paridade bit-exata com os modelos separados.
        self.lilt = LiLTRobertaLikeModel(config, add_pooling_layer=False)        # SER backbone
        self.lilt_re = LiLTRobertaLikeModel(config, add_pooling_layer=False)      # RE backbone
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        hidden_concat_size = (
            config.hidden_size + config.hidden_size // config.channel_shrink_ratio
        )

        # SER head — HF nativo `LiltForTokenClassification` usa só sequence_output (768 dims).
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # RE head — jpWang usa concat (sequence_output, layout_outputs) de dim 960.
        self.extractor = REDecoder(config, hidden_concat_size, use_vectorized=use_re_vectorized)

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
        # KIE-specific
        entities=None,
        relations=None,
        token_span_map=None,
        ser_labels=None,
    ):
        """Forward unificado.

        Argumentos:
        - input_ids, bbox, attention_mask: padrão do LiLT
        - entities: Optional[List[dict{start, end, label}]]. Se None, são extraídas das
            predições do SER (requer `token_span_map`).
        - relations: Optional[List[dict{head, tail}]]. Gold em treinamento. Em inferência,
            passar None faz o REDecoder usar fallback (pares candidatos derivados de entities).
        - token_span_map: List[List[int]]. Para cada amostra do batch, um array do tamanho
            dos tokens (incluindo CLS/SEP/pad) indicando o span_idx de cada token (-1 para
            tokens sem span, como CLS/SEP/pad).
        - ser_labels: Optional gold SER labels para fine-tuning (não usado hoje).

        Retorno: JointKIEOutput.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Passagem 1: backbone do SER → para classifier token-level
        ser_outputs, _ = self.lilt(
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
        ser_seq = ser_outputs[0]  # (B, T, 768)
        ser_logits = self.classifier(self.dropout(ser_seq))

        # Passagem 2: backbone do RE → para extractor
        re_outputs, re_layout = self.lilt_re(
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
        re_seq = re_outputs[0]
        re_input = self.dropout(torch.cat([re_seq, re_layout], -1))

        # Caminho automático: derivar entities das predições SER
        if entities is None:
            if token_span_map is None:
                raise ValueError(
                    "Quando `entities` não é fornecido, é obrigatório passar "
                    "`token_span_map` (List[List[int]] mapeando cada token para span_idx, "
                    "ou -1 para tokens não-span)."
                )
            entities = self._aggregate_ser_to_entities(ser_logits, token_span_map)

        # Em inferência, relations vazio faz o REDecoder popular candidatos automaticamente
        if relations is None:
            relations = [{"head": [], "tail": []} for _ in range(len(entities))]

        # RE head — REDecoder faz build_relation internamente
        loss_re, pred_relations = self.extractor(re_input, entities, relations)

        # Loss SER (só quando ser_labels fornecido; não usado aqui, mas suportado)
        loss = loss_re
        if ser_labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ser_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    ser_labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(ser_labels),
                )
                ser_loss = loss_fct(active_logits, active_labels)
            else:
                ser_loss = loss_fct(ser_logits.view(-1, self.num_labels), ser_labels.view(-1))
            loss = ser_loss + loss_re if loss_re is not None else ser_loss

        return JointKIEOutput(
            loss=loss,
            ser_logits=ser_logits,
            pred_relations=pred_relations,
            entities_used=entities,
        )

    def _aggregate_ser_to_entities(self, ser_logits, token_span_map):
        """Agrega predições SER token-level → entities Q/A por span (majority vote).

        Retorna List[dict{start, end, label}] compatível com REDecoder:
        - label: 0 para QUESTION, 1 para ANSWER (schema do loader XFUN)
        - start, end: índices de tokens no input_ids (inclui CLS/SEP/pad)
        """
        preds = ser_logits.argmax(-1).cpu().tolist()  # (B, T)
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

            # REDecoder precisa de ≥2 entities; senão ele força fallback vazio
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
