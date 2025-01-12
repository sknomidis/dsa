from __future__ import annotations

from typing import Protocol, TypeVar

ValueType = TypeVar("ValueType")


class NotFoundError(Exception):
    pass


class SearchAlgorithm(Protocol):
    """Given a sorted array and a value, return the index of the latter."""

    def __call__(self, array: list[ValueType], value: ValueType) -> int: ...


def linear_search(array: list[ValueType], value: ValueType) -> int:
    """Linear search algorithm.

    Iterate over all elements, until match is found.

    Complexity
    ----------
    Time: O(N)
    Space: O(1)

    Pros
    ----
    - Simple
    - No sorting required
    - No extra memory required

    Cons
    ----
    - Poor performance
    """
    assert list(array) == sorted(array)
    for index, value_stored in enumerate(array):
        if value_stored == value:
            return index
    raise NotFoundError


def binary_search_recursive(array: list[ValueType], value: ValueType) -> int:
    """Recursive binary search algorithm.

    Recursively halve search interval in sorted array, until value is found.

    Complexity
    ----------
    Time: O(log N)
    Space: O(log N)

    Pros
    ----
    - Fast

    Cons
    ----
    - Requires sorting
    """
    assert list(array) == sorted(array)

    def search_subarray(index_start: int, index_end: int) -> int:
        if index_start == index_end:
            raise NotFoundError

        index_middle = (index_start + index_end) // 2
        if array[index_middle] == value:
            # Value has been found
            return index_middle
        if value < array[index_middle]:
            # Search in left subarray
            return search_subarray(index_start, index_middle)
        if value > array[index_middle]:
            # Search in right subarray
            return search_subarray(index_middle + 1, index_end)

    return search_subarray(index_start=0, index_end=len(array))


def binary_search_iterative(array: list[ValueType], value: ValueType) -> int:
    """Iterative binary search algorithm.

    Recursively halve search interval in sorted array, until value is found.

    Complexity
    ----------
    Time: O(log N)
    Space: O(1)

    Pros
    ----
    - Fast

    Cons
    ----
    - Requires sorting
    """
    assert list(array) == sorted(array)

    # Initialization
    index_left = 0
    index_right = len(array)

    while index_left < index_right:
        index_middle = (index_left + index_right) // 2
        if value == array[index_middle]:
            # Value found
            return index_middle

        if value < array[index_middle]:
            # Search in left subarray
            index_right = index_middle
        else:
            # Search in right subarray
            index_left = index_middle + 1
    raise NotFoundError
