from __future__ import annotations

import numpy as np
import pytest

from dsa.algorithms import sorting


@pytest.mark.parametrize(
    "array_input",
    [
        [],
        [42],
        [0, 1],
        [6, 3],
        [2, 0, 1],
        [33, 24, 9, 1],
        np.random.default_rng(seed=42).integers(2**16, size=99).tolist(),
        np.random.default_rng(seed=42).integers(2**16, size=100).tolist(),
    ],
    ids=[
        "0_elements",
        "1_element",
        "2_elements_sorted",
        "2_elements",
        "3_elements",
        "4_elements_reverse",
        "99_elements_random",
        "100_elements_random",
    ],
)
@pytest.mark.parametrize(
    "sorting_algorithm",
    [
        sorting.bubble_sort,
        sorting.heap_sort,
        sorting.insertion_sort,
        sorting.merge_sort_top_down,
        sorting.merge_sort_bottom_up,
        sorting.quicksort_lomuto,
        sorting.quicksort_hoare,
        sorting.selection_sort,
    ],
    ids=lambda x: x.__name__,
)
def test_sorting(array_input: list[int], sorting_algorithm: sorting.SortingAlgorithm) -> None:
    array_expected = sorted(array_input)
    array_returned = sorting_algorithm(list(array_input))
    assert array_returned == array_expected
