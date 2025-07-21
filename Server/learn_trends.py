from __future__ import annotations

"""Scan wordlists and accumulate password pattern statistics in Redis."""

import argparse
from pathlib import Path

from redis_utils import get_redis
from pattern_utils import word_to_pattern, is_valid_word


def process_wordlists(directory: Path) -> None:
    """Parse all files in ``directory`` and increment pattern counts."""
    r = get_redis()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip()
                if not is_valid_word(word):
                    continue
                pattern = word_to_pattern(word)
                r.zincrby("dictionary:patterns", 1, pattern)


def top_patterns(count: int, r=None) -> list[tuple[str, int]]:
    """Return ``count`` most common patterns and their counts."""
    if r is None:
        r = get_redis()
    results = r.zrevrange("dictionary:patterns", 0, count - 1, withscores=True)
    patterns: list[tuple[str, int]] = []
    for pat, score in results:
        if isinstance(pat, bytes):
            pat = pat.decode()
        patterns.append((pat, int(score)))
    return patterns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing wordlists",
    )
    parser.add_argument(
        "--top", type=int, default=0, help="Show N most common patterns after processing",
    )
    args = parser.parse_args()
    process_wordlists(args.directory)
    if args.top:
        for pattern, count in top_patterns(args.top):
            print(f"{pattern} {count}")
