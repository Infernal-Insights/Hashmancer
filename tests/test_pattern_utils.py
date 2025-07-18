import os
import sys
import pytest

# Allow importing the Server package from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Server.pattern_utils import word_to_pattern, is_valid_pattern


def test_word_to_pattern():
    assert word_to_pattern("Dark12!") == "$U$l$l$l$d$d$s"


def test_is_valid_pattern():
    pattern = word_to_pattern("Dark12!")
    assert is_valid_pattern(pattern)
    assert not is_valid_pattern("$Ulldds")
    assert not is_valid_pattern("invalid")
