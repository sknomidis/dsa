from __future__ import annotations

import pytest

from dsa.algorithms import fibonacci


@pytest.mark.parametrize(
    "n, expected",
    [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8), (7, 13), (8, 21), (9, 34), (10, 55), (11, 89), (12, 144)],
    ids=lambda n: f"{n}th",
)
@pytest.mark.parametrize(
    "fibonacci_algorithm",
    [
        fibonacci.fibonacci_brute_force,
        fibonacci.fibonacci_memoization,
        fibonacci.fibonacci_tabulation,
        fibonacci.fibonacci_tabulation_spaced_optimized,
    ],
    ids=lambda f: f.__name__,
)
def test_fibonacci(n: int, expected: int, fibonacci_algorithm: fibonacci.FibonacciAlgorithm) -> None:
    assert fibonacci_algorithm(n) == expected
