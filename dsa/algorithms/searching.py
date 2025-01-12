from __future__ import annotations

from typing import Protocol, Sequence, TypeVar

ValueType = TypeVar("ValueType")


class NotFoundError(Exception):
    pass


class SearchAlgorithm(Protocol):
    """Given a sorted array and a value, return the index of the latter."""

    def __call__(self, array: Sequence[ValueType], value: ValueType) -> int: ...


def linear_search(array: Sequence[ValueType], value: ValueType) -> int:
    """Linear search algorithm.

    Iterate over all elements, until match is found.

    Complexity
    ----------
    Time: O(N)
    Space: O(1)

    Comparison
    ----------
    + Simple
    + No sorting required
    + No extra memory required
    - Poor performance
    """
    assert list(array) == sorted(array)
    for index, value_stored in enumerate(array):
        if value_stored == value:
            return index
    raise NotFoundError


def binary_search_recursive(array: Sequence[ValueType], value: ValueType) -> int:
    """Recursive binary search algorithm.

    Recursively halve search interval in sorted array, until value is found.

    Complexity
    ----------
    Time: O(log N)
    Space: O(log N)

    Comparison
    ----------
    + Fast
    - Requires sorting
    """
    assert list(array) == sorted(array)

    def search_subarray(index_start: int, index_end: int) -> int:
        if index_start == index_end:
            raise NotFoundError

        index_middle = (index_end + index_start) // 2
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


def binary_search_iterative(array: Sequence[ValueType], value: ValueType) -> int:
    """Iterative binary search algorithm.

    Recursively halve search interval in sorted array, until value is found.

    Complexity
    ----------
    Time: O(log N)
    Space: O(1)

    Comparison
    ----------
    + Fast
    - Requires sorting
    """
    assert list(array) == sorted(array)

    # Initialization
    index_left = 0
    index_right = len(array)
    index_middle = (index_left + index_right) // 2

    while value != array[index_middle]:
        if value < array[index_middle]:
            # Keep looking at left subarray
            index_right = index_middle
        else:
            # Keep looking at right subarray
            index_left = index_middle + 1
        index_middle = (index_left + index_right) // 2

        if index_right == 0 or index_left == len(array):
            raise NotFoundError

    return index_middle
