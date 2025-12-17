import re
import unicodedata
from typing import Optional

UNICODE_SPACES = {"Zs", "Zl", "Zp"}

def normalize_text(
    text: Optional[str],
    keep_slash_dash: bool = False,
    keep_dot: bool = False,
) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text).lower()

    # Normalize unicode whitespace (including line/paragraph separators)
    out = []
    for ch in text:
        if unicodedata.category(ch) in UNICODE_SPACES:
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)

    # Remove punctuation, optionally keep separators
    allow = ""
    if keep_slash_dash:
        allow += r"/\-"
    if keep_dot:
        allow += r"\."

    if allow:
        text = re.sub(rf"[^\w\s{allow}]", " ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text
