import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))

from hashmancer.darkling import statistics


def test_probability_order_indices():
    # build simple Markov counts
    records = [("$U$l", 3), ("$U$U", 1)]
    markov = statistics.build_markov(records)
    charset_map = {"?1": "Ac", "?2": "Xy"}
    order = statistics.probability_index_order("?1?2", charset_map, markov)
    assert order[:4] == [1, 0, 2, 3]


def test_probability_order_inverse():
    records = [("$U$l", 3), ("$U$U", 1)]
    markov = statistics.build_markov(records)
    charset_map = {"?1": "Ac", "?2": "Xy"}
    order = statistics.probability_index_order(
        "?1?2", charset_map, markov, inverse=True
    )
    assert order[:4] == [3, 2, 0, 1]


def test_build_markov_validation():
    with pytest.raises(ValueError):
        statistics.build_markov([("abc", 1)])
    with pytest.raises(ValueError):
        statistics.build_markov([("$U$l", -1)])

