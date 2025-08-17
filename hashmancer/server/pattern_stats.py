from __future__ import annotations

"""Build and use password pattern transition statistics stored in Redis."""

import re
import random
from pathlib import Path
from typing import Dict

import redis

from .redis_utils import get_redis
from .pattern_utils import word_to_pattern, is_valid_word

# pattern tokens are produced by word_to_pattern
TOKEN_RE = re.compile(r"\$[Uldsce]")


def update_stats(directory: Path) -> None:
    """Scan files in ``directory`` and accumulate transition counts in Redis."""
    r = get_redis()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip()
                if not is_valid_word(word):
                    continue
                tokens = TOKEN_RE.findall(word_to_pattern(word))
                if not tokens:
                    continue
                r.hincrby("pattern_start", tokens[0], 1)
                for a, b in zip(tokens, tokens[1:]):
                    r.hincrby(f"pattern_trans:{a}", b, 1)


def _weighted_choice(choices: Dict[str, int]) -> str:
    total = sum(choices.values())
    r = random.uniform(0, total)
    upto = 0.0
    for key, weight in choices.items():
        if upto + weight >= r:
            return key
        upto += weight
    return random.choice(list(choices))


def generate_mask(length: int, r: redis.Redis | None = None) -> str:
    """Generate a mask of ``length`` characters using stored statistics."""
    if r is None:
        r = get_redis()
    start_counts = {k: int(v) for k, v in r.hgetall("pattern_start").items()}
    if not start_counts:
        return ""
    token = _weighted_choice(start_counts)
    tokens = [token]
    for _ in range(1, length):
        trans = {k: int(v) for k, v in r.hgetall(f"pattern_trans:{token}").items()}
        if not trans:
            trans = start_counts
        token = _weighted_choice(trans)
        tokens.append(token)
    return "".join(tokens)


def generate_masks(count: int, length: int, r: redis.Redis | None = None) -> list[str]:
    """Return ``count`` generated masks of the given ``length``."""
    if r is None:
        r = get_redis()
    return [generate_mask(length, r) for _ in range(count)]

