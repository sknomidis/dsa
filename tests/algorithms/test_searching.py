from __future__ import annotations

import pytest

from dsa.algorithms import searching


@pytest.mark.parametrize(
    "search_algorithm", [searching.linear_search, searching.binary_search_recursive, searching.binary_search_iterative]
)
def test_search(search_algorithm: searching.SearchAlgorithm):
    array = list(range(0, 100, 4))
    for index, value in enumerate(array):
        assert search_algorithm(array, value) == index
    with pytest.raises(searching.NotFoundError):
        search_algorithm(array, 100)
