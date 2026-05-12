from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.strip())
    return tokens or ["<empty>"]


def detokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""
    out: list[str] = []
    for tok in tokens:
        if tok.startswith("<") and tok.endswith(">"):
            continue
        if out and re.match(r"^[\w<]", tok):
            out.append(" ")
        out.append(tok)
    return "".join(out).strip()
