import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from darkling import charsets


def test_common_symbols_length():
    assert len(charsets.COMMON_SYMBOLS) == 15


def test_emoji_only_symbols():
    assert all(unicodedata.category(c).startswith("S") for c in charsets.EMOJI)
    assert len(charsets.EMOJI) == 40


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
