from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter, XLMRobertaConverter
# PATCH: auto_class_factory removed in transformers>=4.12; use _BaseAutoModelClass instead
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from .models.LiLTRobertaLike import (
    LiLTRobertaLikeConfig,
    LiLTRobertaLikeForRelationExtraction,
    LiLTRobertaLikeForTokenClassification,
    LiLTRobertaLikeTokenizer,
    LiLTRobertaLikeTokenizerFast,
)

CONFIG_MAPPING.update([("liltrobertalike", LiLTRobertaLikeConfig),])
MODEL_NAMES_MAPPING.update([("liltrobertalike", "LiLTRobertaLike"),])
TOKENIZER_MAPPING.update(
    [
        (LiLTRobertaLikeConfig, (LiLTRobertaLikeTokenizer, LiLTRobertaLikeTokenizerFast)),
    ]
)

import os as _os
_tag_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'tag.txt')
with open(_tag_path, 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'
if TAG == 'monolingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": RobertaConverter,})
elif TAG == 'multilingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": XLMRobertaConverter,})

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForTokenClassification),]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForRelationExtraction),]
)

# PATCH: substituição de auto_class_factory por subclasse de _BaseAutoModelClass
class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

class AutoModelForRelationExtraction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_RELATION_EXTRACTION_MAPPING
