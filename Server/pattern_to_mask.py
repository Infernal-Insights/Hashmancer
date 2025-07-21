from __future__ import annotations
"""Convert stored password patterns into hashcat-style masks."""

import argparse
from typing import List

from redis_utils import get_redis
from pattern_stats import TOKEN_RE

TOKEN_TO_MASK = {
    "$U": "?u",
    "$l": "?l",
    "$d": "?d",
    "$c": "?s",
    "$s": "?a",
    "$e": "?a",
}

def pattern_to_mask(pattern: str) -> str:
    """Translate a stored pattern to a hashcat mask string."""
    tokens = TOKEN_RE.findall(pattern)
    return "".join(TOKEN_TO_MASK.get(t, "?a") for t in tokens)


def get_top_masks(count: int, key: str = "dictionary:patterns", r=None) -> List[str]:
    """Return ``count`` masks generated from the most common patterns."""
    if r is None:
        r = get_redis()
    raw = r.zrevrange(key, 0, count - 1)
    masks: List[str] = []
    for item in raw:
        pattern = item.decode() if isinstance(item, bytes) else item
        masks.append(pattern_to_mask(pattern))
    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Number of masks to output")
    parser.add_argument("--key", default="dictionary:patterns", help="Redis key containing patterns")
    args = parser.parse_args()
    for mask in get_top_masks(args.count, args.key):
        print(mask)
