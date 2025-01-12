from __future__ import annotations

from typing import Protocol


class FibonacciAlgorithm(Protocol):
    def __call__(self, n: int) -> int: ...


def fibonacci_brute_force(n: int) -> int:
    """Brute-force recursive approach.

    Complexity
    ----------
    Time: O(2**N)
    Space: O(N)
    """
    assert n >= 0
    if n < 2:
        return n
    return fibonacci_brute_force(n - 1) + fibonacci_brute_force(n - 2)


def fibonacci_memoization(n: int) -> int:
    """Memoization recursive approach (top-down storage).

    In memoization we store the output of function calls in a table (recursion).

    Complexity
    ----------
    Time: O(N)
    Space: O(N)
    """
    table = (n + 1) * [None]

    def fibonacci_cached(n: int) -> int:
        assert n >= 0
        if n < 2:
            return n
        if table[n] is None:
            table[n] = fibonacci_cached(n - 1) + fibonacci_cached(n - 2)
        return table[n]

    return fibonacci_cached(n)


def fibonacci_tabulation(n: int) -> int:
    """Tabulation iterative approach (bottom-up storage).

    In tabulation we store the output of subproblems in a table (iteration).

    Complexity
    ----------
    Time: O(N)
    Space: O(N)
    """
    assert n >= 0
    if n < 2:
        return n
    table = [None if m > 2 else m for m in range(n + 1)]
    for m in range(2, n + 1):
        table[m] = table[m - 1] + table[m - 2]
    return table[m]


def fibonacci_tabulation_spaced_optimized(n: int) -> int:
    """Space-optimized tabulation iterative approach (bottom-up storage).

    Complexity
    ----------
    Time: O(N)
    Space: O(1)
    """
    assert n >= 0
    if n < 2:
        return n
    value_1, value_2 = 0, 1
    for _ in range(2, n + 1):
        current = value_1 + value_2
        value_1, value_2 = value_2, current
    return current
