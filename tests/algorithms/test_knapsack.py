from __future__ import annotations

import pytest

from dsa.algorithms import knapsack


@pytest.mark.parametrize(
    "knapsack_algorithm",
    [knapsack.knapsack_brute_force, knapsack.knapsack_memoization, knapsack.knapsack_tabulation],
    ids=lambda f: f.__name__,
)
def test_knapsack(knapsack_algorithm: knapsack.KnapsackAlgorithm) -> None:
    assert knapsack_algorithm(weights_and_profits=[(4, 1), (5, 2), (1, 3)], capacity=4) == 3
    assert knapsack_algorithm(weights_and_profits=[(4, 1), (5, 2), (6, 3)], capacity=3) == 0
    assert knapsack_algorithm(weights_and_profits=[(2, 1), (5, 8), (1, 2)], capacity=4) == 3
    assert knapsack_algorithm(weights_and_profits=[(2, 1), (5, 8), (1, 2)], capacity=5) == 8
    assert knapsack_algorithm(weights_and_profits=[(2, 1), (5, 8), (1, 2)], capacity=6) == 10
    assert knapsack_algorithm(weights_and_profits=[(5, 10), (4, 40), (6, 30), (3, 50)], capacity=5) == 50
