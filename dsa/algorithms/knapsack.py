from __future__ import annotations

from typing import Protocol, Sequence, TypeAlias

WeightAndProfitType: TypeAlias = tuple[int, int]


class KnapsackAlgorithm(Protocol):
    """Solution to the knapsack problem.

    Given a set of items, each with a weight and a value, determine which items
    to include in the collection, so that the total weight is less than or equal
    to a given limit, and the total value is as large as possible.
    """

    def __call__(self, weights_and_profits: Sequence[WeightAndProfitType], capacity: int) -> int: ...


def knapsack_brute_force(weights_and_profits: Sequence[WeightAndProfitType], capacity: int) -> int:
    """Brute-force recursive solution to knapsack problem.

    Consider all subsets of items, and compute the total weight and profit for
    each case. Out of those with weight smaller than `capacity`, pick the one
    with higher profit.

    Complexity
    ----------
    Time: O(2**N)
    Space: O(N)
    """

    def compute_max_profit(n_items_left: int, capacity_left: int) -> int:
        if n_items_left == 0 or capacity_left == 0:
            # Nothing more to do
            return 0
        # Pick next item
        weight_item, profit_item = weights_and_profits[n_items_left - 1]
        if weight_item > capacity_left:
            # Item cannot fit, so it gets dropped
            return compute_max_profit(n_items_left - 1, capacity_left)
        # Decide if it is profitable to include item or not
        profits_item_included = compute_max_profit(n_items_left - 1, capacity_left - weight_item) + profit_item
        profits_item_excluded = compute_max_profit(n_items_left - 1, capacity_left)
        return max(profits_item_included, profits_item_excluded)

    return compute_max_profit(n_items_left=len(weights_and_profits), capacity_left=capacity)


def knapsack_memoization(weights_and_profits: Sequence[WeightAndProfitType], capacity: int) -> int:
    """Memoization recursive solution to knapsack problem (top-down).

    Complexity
    ----------
    Time: O(N x capacity)
    Space: O(N x capacity)
    """
    table = [[None for _ in range(capacity + 1)] for _ in range(len(weights_and_profits) + 1)]

    def compute_max_profit(n_items_left: int, capacity_left: int) -> int:
        if table[n_items_left][capacity_left] is not None:
            # Already computed, return cache
            return table[n_items_left][capacity_left]
        if n_items_left == 0 or capacity_left == 0:
            # Nothing more to do
            table[n_items_left][capacity_left] = 0
            return table[n_items_left][capacity_left]
        # Pick next item
        weight_item, profit_item = weights_and_profits[n_items_left - 1]
        if weight_item > capacity_left:
            # Item cannot fit, so it gets dropped
            table[n_items_left][capacity_left] = compute_max_profit(n_items_left - 1, capacity_left)
            return table[n_items_left][capacity_left]
        # Decide if it is profitable to include item or not
        profits_item_included = compute_max_profit(n_items_left - 1, capacity_left - weight_item) + profit_item
        profits_item_excluded = compute_max_profit(n_items_left - 1, capacity_left)
        table[n_items_left][capacity_left] = max(profits_item_included, profits_item_excluded)
        return table[n_items_left][capacity_left]

    return compute_max_profit(n_items_left=len(weights_and_profits), capacity_left=capacity)


def knapsack_tabulation(weights_and_profits: Sequence[WeightAndProfitType], capacity: int) -> int:
    """Tabulation iterative solution to knapsack problem (bottom-up).

    Complexity
    ----------
    Time: O(N x capacity)
    Space: O(N x capacity)
    """
    n_items = len(weights_and_profits)
    table = [[None for _ in range(capacity + 1)] for _ in range(n_items + 1)]
    for n_items_left in range(n_items + 1):
        for capacity_left in range(capacity + 1):
            if n_items_left == 0 or capacity_left == 0:
                # Nothing more to do
                table[n_items_left][capacity_left] = 0
                continue
            weight_item, profit_item = weights_and_profits[n_items_left - 1]
            if weight_item > capacity_left:
                # Item cannot fit, so it gets dropped
                table[n_items_left][capacity_left] = table[n_items_left - 1][capacity_left]
                continue
            # Decide if it is profitable to include item or not
            profits_item_included = table[n_items_left - 1][capacity_left - weight_item] + profit_item
            profits_item_excluded = table[n_items_left - 1][capacity_left]
            table[n_items_left][capacity_left] = max(profits_item_included, profits_item_excluded)
    return table[n_items][capacity]
