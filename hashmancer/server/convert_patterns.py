from __future__ import annotations

"""Convert stored password patterns to the latest token scheme."""

import argparse

from .redis_utils import get_redis
from .pattern_utils import word_to_pattern


def convert(src_key: str, dest_key: str) -> None:
    """Read patterns from ``src_key`` and write updated tokens to ``dest_key``."""
    r = get_redis()
    pipe = r.pipeline()
    processed = 0
    for pattern, score in r.zscan_iter(src_key):
        new_pattern = word_to_pattern(pattern)
        pipe.zincrby(dest_key, float(score), new_pattern)
        processed += 1
        if processed % 1000 == 0:
            pipe.execute()
    pipe.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="dictionary:patterns", help="Source key")
    parser.add_argument("--dest", default="dictionary:patterns:v2", help="Destination key")
    args = parser.parse_args()
    convert(args.src, args.dest)
