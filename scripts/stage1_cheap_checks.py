import re


def tokenize(text):
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap_ratio(claim, passage):
    c = tokenize(claim)
    p = tokenize(passage)
    if not c:
        return 0.0
    return len(set(c) & set(p)) / len(set(c))


def score_disagreement(scores, gap=0.2, std=0.1):
    if not scores:
        return False
    arr = [float(s) for s in scores if s is not None]
    if len(arr) < 2:
        return False
    mx = max(arr)
    mn = min(arr)
    mean = sum(arr) / len(arr)
    var = sum((x - mean) ** 2 for x in arr) / len(arr)
    return (mx - mn) > gap or var ** 0.5 > std


def apply_cheap_checks(decision, claim, passages, scores=None, overlap_min=0.1, gap=0.2, std=0.1):
    if decision not in ("ACCEPT", "REJECT"):
        return decision, "unchanged"
    reasons = []
    if claim is not None and passages:
        top_passage = passages[0] if isinstance(passages, list) else passages
        if overlap_ratio(claim, top_passage) < overlap_min:
            reasons.append("low_overlap")
    if scores is not None and score_disagreement(scores, gap=gap, std=std):
        reasons.append("score_disagreement")
    if reasons:
        return "UNCERTAIN", ",".join(reasons)
    return decision, "unchanged"
