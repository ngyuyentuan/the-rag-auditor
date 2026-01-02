from typing import Literal, Tuple


FinalDecision = Literal["ACCEPT", "REJECT", "UNCERTAIN"]


def finalize_decision(
    stage1_decision: str,
    stage2: dict,
    *,
    entail_min: float,
    contradict_min: float,
    neutral_max: float,
) -> Tuple[FinalDecision, str]:
    if stage1_decision in {"ACCEPT", "REJECT"} and not stage2.get("ran", False):
        return stage1_decision, "stage1_only"
    if stage2.get("ran", False):
        nli = stage2.get("nli", {})
        label_probs = nli.get("label_probs")
        if isinstance(label_probs, dict):
            p_entail = float(label_probs.get("ENTAILMENT", 0.0))
            p_contra = float(label_probs.get("CONTRADICTION", 0.0))
            p_neutral = float(label_probs.get("NEUTRAL", 0.0))
            if p_entail >= entail_min and p_neutral <= neutral_max:
                return "ACCEPT", "stage2_entailment"
            if p_contra >= contradict_min and p_neutral <= neutral_max:
                return "REJECT", "stage2_contradiction"
            return "UNCERTAIN", "stage2_ambiguous"
        return "UNCERTAIN", "stage2_missing_nli"
    return "UNCERTAIN", "stage2_not_run"
