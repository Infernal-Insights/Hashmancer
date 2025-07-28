#!/usr/bin/env python3
"""Simple launcher for the darkling demo kernel."""
import argparse
import subprocess
from darkling import charsets


def get_lang_sets(name: str) -> tuple[str, str]:
    key = name.replace("-", "_").upper()
    upper = getattr(charsets, f"{key}_UPPER", charsets.ENGLISH_UPPER)
    lower = getattr(charsets, f"{key}_LOWER", charsets.ENGLISH_LOWER)
    return upper, lower


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch darkling-host with language charsets")
    parser.add_argument("--lang", default="English", help="language for ?1/?2 alphabets")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="additional options for darkling-host")
    args = parser.parse_args()

    upper, lower = get_lang_sets(args.lang)
    cmd = ["./darkling-host", "-1", upper, "-2", lower] + args.extra
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

