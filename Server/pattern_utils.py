import re

# Token regex for mask patterns like $U$l$d$s
MASK_RE = re.compile(r'(?:\$(?:U|l|d|s))+')


def word_to_pattern(word: str) -> str:
    """Return a mask pattern representing the character classes in *word*."""
    tokens: list[str] = []
    for ch in word:
        if ch.isupper():
            tokens.append('$U')
        elif ch.islower():
            tokens.append('$l')
        elif ch.isdigit():
            tokens.append('$d')
        else:
            tokens.append('$s')
    return ''.join(tokens)


def is_valid_pattern(pattern: str) -> bool:
    """Return True if *pattern* is a valid mask pattern."""
    return bool(MASK_RE.fullmatch(pattern))
