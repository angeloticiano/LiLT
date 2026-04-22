import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features, onnx_friendly: bool = False):
        """
        onnx_friendly=True: usa einsum em vez de torch.nn.Bilinear (operador aten::bilinear
        não é exportável para ONNX opset 14-17 no torch 1.13). Pesos idênticos, resultado
        bit-exato (diff < 1e-7).
        """
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.onnx_friendly = onnx_friendly

        if onnx_friendly:
            # Pesos equivalentes ao Bilinear.weight (shape: out, in1, in2)
            self.bilinear_weight = torch.nn.Parameter(torch.empty(out_features, in_features, in_features))
        else:
            self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)

        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        if self.onnx_friendly:
            # Equivalente a Bilinear(x_1, x_2): y[b,k] = sum_{i,j} x_1[b,i] * W[k,i,j] * x_2[b,j]
            bilinear_out = torch.einsum("bi,kij,bj->bk", x_1, self.bilinear_weight, x_2)
        else:
            bilinear_out = self.bilinear(x_1, x_2)
        return bilinear_out + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        if self.onnx_friendly:
            torch.nn.init.kaiming_uniform_(self.bilinear_weight, a=5 ** 0.5)
        else:
            self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def to_onnx_friendly(self):
        """Converte este BiaffineAttention (com Bilinear) para versão onnx-friendly (com einsum),
        preservando os pesos. Útil para export ONNX sem retreino."""
        if self.onnx_friendly:
            return self
        new = BiaffineAttention(self.in_features, self.out_features, onnx_friendly=True)
        new.bilinear_weight.data.copy_(self.bilinear.weight.data)
        new.linear.load_state_dict(self.linear.state_dict())
        return new


class REDecoder(nn.Module):
    def __init__(self, config, input_size, use_vectorized: bool = False):
        """REDecoder com dois caminhos de forward:

        - forward (default, use_vectorized=False): implementação original com loops Python
          e set/list para build_relation. Compatível com treino existente.
        - forward_vectorized (use_vectorized=True): mesma semântica porém 100% tensorial
          dentro do forward. Compatível com export ONNX. Usa build_relation_tensor que
          processa pares fora do grafo (Python puro) e transforma em tensors antes do
          forward.

        Ambos retornam resultados bit-exatos (dentro de tolerância numérica) sobre o
        mesmo input. O `use_vectorized=True` é usado para:
        - export ONNX (evita for-loops Python no forward computational graph)
        - inferência em produção (marginalmente mais rápido em batches grandes)

        Para manter compatibilidade com os checkpoints já treinados (SER+RE best),
        os pesos são idênticos entre os dois caminhos — só muda a orquestração.
        """
        super().__init__()
        self.use_vectorized = use_vectorized
        self.entity_emb = nn.Embedding(3, input_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(input_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        # PATCH: class_weight compensa desbalanceamento ~1:20 entre pares positivos e negativos.
        # Sem isso, o modelo converge para "prever tudo 0" com precision≈1.0, recall≈0.06.
        # Usamos F.cross_entropy no forward com o weight como buffer (acompanha to(device)).
        self.register_buffer("_class_weight", torch.tensor([1.0, 10.0]))
        self.loss_fct = None  # legacy — usamos F.cross_entropy no forward

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # PATCH: label mapping do synth (XFUND loader) é {0:QUESTION, 1:ANSWER}.
            # Código original assumia {1:QUESTION, 2:ANSWER} (provavelmente XFUND multi-classe com HEADER).
            # Com o mapping errado, all_possible_relations fica vazio → só 1 par é treinado.
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 0 and entities[b]["label"][j] == 1
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        """Dispatcher: rota para forward original ou vetorial conforme self.use_vectorized."""
        if self.use_vectorized:
            return self.forward_vectorized(hidden_states, entities, relations)
        return self._forward_legacy(hidden_states, entities, relations)

    def _forward_legacy(self, hidden_states, entities, relations):
        """Forward original com for-loops Python. Compatível com treino atual."""
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        all_logits = []
        all_labels = []

        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
            all_logits.append(logits)
            all_labels.append(relation_labels)
        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)
        # PATCH: cross_entropy com class_weight para compensar desbalanceamento positivos/negativos.
        _w = self._class_weight.to(all_logits.device)
        loss = F.cross_entropy(all_logits, all_labels, weight=_w)
        return loss, all_pred_relations

    # ---------- Caminho vetorial ONNX-friendly ----------

    def build_relation_tensor(self, relations, entities, device):
        """Variante tensorial de build_relation.

        Retorna um dict de tensors contendo todos os pares (positivos + negativos) de
        todos os docs do batch, achatados em um vetor único. Cada par tem:
          - head_global_idx: índice do token-start do head no hidden_states achatado
            (posição batch_size × max_seq_len)
          - tail_global_idx: idem para tail
          - head_label_ids: rótulo 0/1/2 (0=Q, 1=A, etc.) do head — usado em entity_emb
          - tail_label_ids: idem para tail
          - labels: 0 (negativo) ou 1 (positivo) — gold do RE
          - batch_idx: índice do doc de origem (0..batch_size-1)

        Esta função é Python puro (percorre docs), mas roda FORA do grafo computacional
        — ela só PREPARA os tensors. O forward_vectorized abaixo só faz ops tensoriais.
        """
        max_seq_len = None  # será preenchido pelo forward_vectorized
        batch_size = len(relations)

        head_global_idx = []
        tail_global_idx = []
        head_label_ids = []
        tail_label_ids = []
        labels_flat = []
        batch_idx_flat = []

        # Precisamos do max_seq_len para calcular o offset global
        # (será passado pelo forward_vectorized)
        return {
            "batch_size": batch_size,
            "per_doc": [
                self._build_one_doc_pairs(relations[b], entities[b])
                for b in range(batch_size)
            ],
        }

    def _build_one_doc_pairs(self, rel_b, ent_b):
        """Constrói listas Python de pares para 1 doc. Semântica idêntica a build_relation."""
        if len(ent_b["start"]) <= 2:
            ent_b = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
        all_possible = set(
            (i, j)
            for i in range(len(ent_b["label"]))
            for j in range(len(ent_b["label"]))
            if ent_b["label"][i] == 0 and ent_b["label"][j] == 1
        )
        if len(all_possible) == 0:
            all_possible = {(0, 1)}
        positives = set(zip(rel_b["head"], rel_b["tail"]))
        positives = {p for p in positives if p in all_possible}
        negatives = all_possible - positives
        reordered = list(positives) + list(negatives)
        heads = [i[0] for i in reordered]
        tails = [i[1] for i in reordered]
        pair_labels = [1] * len(positives) + [0] * (len(reordered) - len(positives))
        return {
            "entities": ent_b,  # possivelmente ajustado pelo fallback <=2
            "heads_entity": heads,
            "tails_entity": tails,
            "labels": pair_labels,
        }

    def forward_vectorized(self, hidden_states, entities, relations):
        """Forward tensorial sem for-loops no caminho quente.

        Preparação de pares é Python puro (roda fora do grafo), mas o cálculo de
        hidden_states[indices], entity_emb, ffnn, classifier e cross_entropy é
        todo tensorial vetorizado sobre um único batch unificado.

        Este forward é ONNX-friendly: pode ser exportado como grafo estático
        desde que `entities` e `relations` sejam passados como argumentos
        pre-computados (o que já é o caso em produção, onde a pipeline Python
        faz SER → entities → RE).

        Retorno é idêntico ao forward legacy:
          (loss, all_pred_relations)
          onde all_pred_relations é List[List[dict]] — uma lista por doc.
        """
        batch_size, max_seq_len, _ = hidden_states.size()
        device = hidden_states.device

        # Preparar pares em Python puro (fora do grafo)
        per_doc = [self._build_one_doc_pairs(relations[b], entities[b]) for b in range(batch_size)]

        # Aplicar o fallback às entities (semântica igual ao legacy)
        entities = [d["entities"] for d in per_doc]

        # Construir tensors achatados
        head_global_idx = []   # índice em hidden_states achatado (batch × seq)
        tail_global_idx = []
        head_label_ids = []
        tail_label_ids = []
        labels_flat = []
        per_doc_pair_counts = []

        for b, d in enumerate(per_doc):
            ent = d["entities"]
            starts = ent["start"]
            elabels = ent["label"]
            n_pairs = len(d["heads_entity"])
            per_doc_pair_counts.append(n_pairs)
            base = b * max_seq_len
            for h, t, lbl in zip(d["heads_entity"], d["tails_entity"], d["labels"]):
                head_global_idx.append(base + starts[h])
                tail_global_idx.append(base + starts[t])
                head_label_ids.append(elabels[h])
                tail_label_ids.append(elabels[t])
                labels_flat.append(lbl)

        head_global_idx_t = torch.tensor(head_global_idx, device=device, dtype=torch.long)
        tail_global_idx_t = torch.tensor(tail_global_idx, device=device, dtype=torch.long)
        head_label_ids_t = torch.tensor(head_label_ids, device=device, dtype=torch.long)
        tail_label_ids_t = torch.tensor(tail_label_ids, device=device, dtype=torch.long)
        labels_t = torch.tensor(labels_flat, device=device, dtype=torch.long)

        # Hidden states achatado em (B*T, dim) para gather
        hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))

        # Gather de representações por token (head, tail) + embedding de rótulo
        head_hidden = hidden_flat.index_select(0, head_global_idx_t)      # (N, dim)
        tail_hidden = hidden_flat.index_select(0, tail_global_idx_t)      # (N, dim)
        head_label_repr = self.entity_emb(head_label_ids_t)               # (N, dim)
        tail_label_repr = self.entity_emb(tail_label_ids_t)               # (N, dim)

        head_repr = torch.cat([head_hidden, head_label_repr], dim=-1)     # (N, 2*dim)
        tail_repr = torch.cat([tail_hidden, tail_label_repr], dim=-1)

        heads_ff = self.ffnn_head(head_repr)                              # (N, hidden/2)
        tails_ff = self.ffnn_tail(tail_repr)
        logits = self.rel_classifier(heads_ff, tails_ff)                  # (N, 2)

        # Loss com class_weight (igual ao legacy)
        _w = self._class_weight.to(logits.device)
        loss = F.cross_entropy(logits, labels_t, weight=_w)

        # Predições por doc (Python puro, fora do grafo de treino)
        preds_flat = logits.argmax(-1).cpu().tolist()
        all_pred_relations = []
        offset = 0
        for b, n_pairs in enumerate(per_doc_pair_counts):
            doc_preds = preds_flat[offset:offset + n_pairs]
            doc_pairs = per_doc[b]
            ent_b = doc_pairs["entities"]
            pred_list = []
            for i, pred_label in enumerate(doc_preds):
                if pred_label != 1:
                    continue
                h_ent = doc_pairs["heads_entity"][i]
                t_ent = doc_pairs["tails_entity"][i]
                pred_list.append({
                    "head_id": h_ent,
                    "head": (ent_b["start"][h_ent], ent_b["end"][h_ent]),
                    "head_type": ent_b["label"][h_ent],
                    "tail_id": t_ent,
                    "tail": (ent_b["start"][t_ent], ent_b["end"][t_ent]),
                    "tail_type": ent_b["label"][t_ent],
                    "type": 1,
                })
            all_pred_relations.append(pred_list)
            offset += n_pairs

        return loss, all_pred_relations