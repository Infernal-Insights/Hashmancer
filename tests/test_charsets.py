import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(__file__))

from hashmancer.darkling import charsets


def test_common_symbols_length():
    assert len(charsets.COMMON_SYMBOLS) == 15


def test_emoji_only_symbols():
    assert all(unicodedata.category(c).startswith("S") for c in charsets.EMOJI)
    assert len(charsets.EMOJI) == 40


def test_digits_constant():
    assert charsets.DIGITS == "0123456789"
    assert len(charsets.DIGITS) == 10


def test_ascii_upper_lower_constant():
    assert (
        charsets.ASCII_UPPER_LOWER
        == charsets.ENGLISH_UPPER + charsets.ENGLISH_LOWER
    )


def test_german_umlauts():
    assert "Ä" in charsets.GERMAN_UPPER
    assert "ß" in charsets.GERMAN_LOWER


def test_french_cedilla():
    assert "ç" in charsets.FRENCH_LOWER


def test_arabic_has_no_ascii():
    assert not any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in charsets.ARABIC)


def test_chinese_contains_common_character():
    assert "的" in charsets.CHINESE


def test_japanese_contains_scripts():
    assert "あ" in charsets.JAPANESE
    assert "ア" in charsets.JAPANESE


def test_all_language_alphabets_defined():
    langs = [n[:-6] for n in dir(charsets) if n.endswith("_UPPER")]
    for lang in langs:
        upper = getattr(charsets, f"{lang}_UPPER")
        lower = getattr(charsets, f"{lang}_LOWER")
        assert upper
        assert lower
