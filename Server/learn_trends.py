from __future__ import annotations

"""Scan wordlists and accumulate password pattern statistics in Redis."""

import argparse
from pathlib import Path

from redis_utils import get_redis
from pattern_utils import word_to_pattern


def process_wordlists(directory: Path) -> None:
    """Parse all files in ``directory`` and increment pattern counts."""
    r = get_redis()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        with path.open("r", errors="ignore") as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue
                pattern = word_to_pattern(word)
                r.zincrby("dictionary:patterns", 1, pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing wordlists",
    )
    args = parser.parse_args()
    process_wordlists(args.directory)
