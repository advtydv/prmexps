from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Step:
    """A single reasoning step extracted from a chain."""

    index: int
    text: str


_NUMBERED_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Step\s+)?(\d+)\s*[.:)]\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\((\d+)\)\s*"),
]


def split_reasoning_chain(chain: str) -> list[Step]:
    """Split a reasoning chain string into a list of Step objects.

    Tries numbered patterns first, falls back to double-newline splitting,
    then sentence-level splitting.
    """
    chain = chain.strip()
    if not chain:
        return []

    for pattern in _NUMBERED_PATTERNS:
        steps = _split_by_pattern(chain, pattern)
        if len(steps) > 1:
            return steps

    steps = _split_by_double_newline(chain)
    if len(steps) > 1:
        return steps

    steps = _split_by_sentences(chain)
    if len(steps) > 1:
        return steps

    return [Step(index=0, text=chain)]


def _split_by_pattern(chain: str, pattern: re.Pattern) -> list[Step]:
    splits = pattern.split(chain)

    parts: list[str] = []
    i = 0
    if splits[0].strip():
        parts.append(splits[0].strip())
        i = 1
    else:
        i = 1

    while i < len(splits):
        if i + 1 < len(splits):
            parts.append(splits[i + 1].strip())
            i += 2
        else:
            if splits[i].strip():
                parts.append(splits[i].strip())
            i += 1

    return [Step(index=idx, text=t) for idx, t in enumerate(parts) if t]


def _split_by_double_newline(chain: str) -> list[Step]:
    parts = re.split(r"\n\s*\n", chain)
    parts = [p.strip() for p in parts if p.strip()]
    return [Step(index=idx, text=t) for idx, t in enumerate(parts)]


def _split_by_sentences(chain: str) -> list[Step]:
    parts = re.split(r"(?<=[.!?])\s+", chain)
    parts = [p.strip() for p in parts if p.strip()]
    return [Step(index=idx, text=t) for idx, t in enumerate(parts)]
