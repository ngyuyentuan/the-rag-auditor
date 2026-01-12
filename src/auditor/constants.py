from enum import Enum


class Verdict(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEI = "NEI"
    
    def __str__(self) -> str:
        return self.value


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class ThresholdConfig:
    ENTAIL_HIGH = 0.60
    ENTAIL_MED = 0.45
    CONTRA_HIGH = 0.55
    CONTRA_MED = 0.40
    NEUTRAL_HIGH = 0.50
    
    REFUTES_BOOST_MAX = 0.40
    SUPPORTS_BOOST_MAX = 0.40
    
    SAME_TEXT_THRESHOLD = 1.2


class PatternWeights:
    ONLY_NATIONALITY = 0.30
    EXCLUSIVELY = 0.25
    NOT_PHRASE = 0.18
    CONTRACTION = 0.18
    NEVER = 0.22
    INCAPABLE = 0.22
    YET_TO = 0.18
    WITHOUT = 0.12
    
    YEAR_MATCH = 0.15
    ENTITY_OVERLAP = 0.08
    PHRASE_MATCH = 0.05
    VERB_MATCH = 0.05


NLI_MODEL_DEFAULT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMBEDDER_MODEL_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_EVIDENCE_LENGTH = 1500
MAX_CLAIM_LENGTH = 500
MAX_TOKENIZER_LENGTH = 512
