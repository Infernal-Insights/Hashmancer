from __future__ import annotations

"""Utilities for probabilistic candidate ordering for the darkling engine."""

from collections import defaultdict
from typing import Iterable, Dict, List

from Server.redis_utils import get_redis
from Server.pattern_stats import TOKEN_RE
from darkling import charsets


def _char_token(ch: str) -> str:
    if ch.isupper():
        return "$U"
    if ch.islower():
        return "$l"
    if ch.isdigit():
        return "$d"
    if ch in charsets.COMMON_SYMBOLS:
        return "$c"
    if ch in charsets.EMOJI:
        return "$e"
    return "$s"


def build_markov(records: Iterable[tuple[str, int]]) -> dict:
    """Return start and transition counts from pattern/count pairs."""
    start: Dict[str, int] = defaultdict(int)
    trans: List[Dict[str, Dict[str, int]]] = []
    for pattern, count in records:
        tokens = TOKEN_RE.findall(pattern)
        if not tokens:
            continue
        start[tokens[0]] += int(count)
        for i in range(1, len(tokens)):
            while len(trans) < i:
                trans.append(defaultdict(lambda: defaultdict(int)))
            trans[i - 1][tokens[i - 1]][tokens[i]] += int(count)
    # convert nested defaultdicts to normal dicts
    out_trans: List[Dict[str, Dict[str, int]]] = []
    for pos in trans:
        pos_dict: Dict[str, Dict[str, int]] = {}
        for prev, nxt in pos.items():
            pos_dict[prev] = dict(nxt)
        out_trans.append(pos_dict)
    return {"start": dict(start), "trans": out_trans}


def load_markov(r=None) -> dict:
    """Load pattern statistics from Redis and build Markov tables."""
    if r is None:
        r = get_redis()
    pairs = r.zrange("dictionary:patterns", 0, -1, withscores=True)
    return build_markov((p, int(s)) for p, s in pairs)


def _digits_to_index(digits: List[int], bases: List[int]) -> int:
    idx = 0
    for d, b in zip(digits, bases):
        idx = idx * b + d
    return idx


def _sorted_indices(charset: str, probs: Dict[str, int]) -> List[int]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, ch in enumerate(charset):
        groups[_char_token(ch)].append(i)
    tokens = sorted(groups, key=lambda t: probs.get(t, 0), reverse=True)
    order: List[int] = []
    for t in tokens:
        order.extend(groups[t])
    return order


def probability_index_order(mask: str, charset_map: Dict[str, str], markov: dict, limit: int | None = None) -> List[int]:
    """Return candidate indices for *mask* sorted by Markov probability."""
    charsets_list: List[str] = []
    i = 0
    while i < len(mask):
        if mask[i] == "?" and i + 1 < len(mask) and mask[i + 1].isdigit():
            key = mask[i:i+2]
            charsets_list.append(charset_map.get(key, ""))
            i += 2
        else:
            charsets_list.append(mask[i])
            i += 1
    bases = [len(cs) if len(cs) > 0 else 1 for cs in charsets_list]

    start_probs = markov.get("start", {})
    trans = markov.get("trans", [])
    digits = [0] * len(charsets_list)
    indices: List[int] = []

    def recurse(pos: int, prev: str):
        nonlocal indices
        if limit is not None and len(indices) >= limit:
            return
        if pos == len(charsets_list):
            indices.append(_digits_to_index(digits, bases))
            return
        charset = charsets_list[pos]
        probs = start_probs if pos == 0 else trans[pos - 1].get(prev, start_probs)
        for idx in _sorted_indices(charset, probs):
            digits[pos] = idx
            token = _char_token(charset[idx]) if charset else prev
            recurse(pos + 1, token)
            if limit is not None and len(indices) >= limit:
                return

    recurse(0, "")
    return indices
