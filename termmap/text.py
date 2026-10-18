"""Text normalization shared by queries and dictionary descriptions."""
import re

_non_alnum = re.compile(r"[^a-z0-9+]+")
_spaces = re.compile(r"\s+")


def clean(text) -> str:
    """lowercase; keep letters, digits and '+'; collapse whitespace."""
    t = str(text or "").lower()
    t = t.replace("<br>", " ")
    t = _non_alnum.sub(" ", t)
    return _spaces.sub(" ", t).strip()


def expand(text: str, abbrev: dict) -> str:
    """Token-level abbreviation expansion (1- and 2-token keys)."""
    if not abbrev:
        return text
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        two = " ".join(toks[i:i + 2])
        if i + 1 < len(toks) and two in abbrev:
            out.append(abbrev[two]); i += 2; continue
        out.append(abbrev.get(toks[i], toks[i])); i += 1
    return " ".join(out)
