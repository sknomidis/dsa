"""A selection of well-known sorting algorithms."""

from __future__ import annotations

import heapq
from typing import Callable, Protocol, TypeVar

ValueType = TypeVar("ValueType")


class SortingAlgorithm(Protocol):
    """Place elements in the given list in ascending order."""

    def __call__(self, array: list[ValueType]) -> list[ValueType]:
        pass


def bubble_sort(array: list[ValueType]) -> list[ValueType]:
    """One of the simplest and worst-performing sorting algorithms.

    Adjacent elements are repeatedly swapped if they are in the wrong order.
    Large elements bubble up towards the end of the array, hence the name of
    the algorithm.

    Advantages:
    - Simple
    - Stable
    - Low memory

    Disadvantages:
    - Too slow for large input

    Complexity
    ----------
    Time:  O(N**2)
    Space: O(1)
    """
    # Progressively sort elements
    for _ in range(len(array)):
        # Go through unsorted array and put adjacent elements in order
        for index_bubble in range(len(array) - 1):
            if array[index_bubble] > array[index_bubble + 1]:
                _swap_elements(array, index_bubble, index_bubble + 1)
    return array


def heap_sort(array: list[ValueType]) -> list[ValueType]:
    """Sorting based on the heap data structure.

    Logic
    -----
    Construct a min heap, and iteratively extract smallest element.

    Advantages:
    - Good worst-case performance
    - Low memory

    Disadvantages:
    - Not stable
    - Slower than merge sort

    Complexity
    ----------
    Time:  O(N log N)
    Space: O(1)
    """
    # TODO: Use custom heap implementation
    # Construct heap
    heap = []
    for _ in range(len(array)):
        heapq.heappush(heap, array.pop())
    # Successively extract minimum elements
    return [heapq.heappop(array) for _ in range(len(array))]


def insertion_sort(array: list[ValueType]) -> list[ValueType]:
    """Simple sorting algorithm that builds sorted array one item at a time.

    Logic
    -----
    It progressively constructs a sorted portion of the array, by iteratively
    inserting each element into its right position.

    Complexity
    ----------
    Time:  O(N**2)
    Space: O(1)
    """
    array_output = list(array).copy()

    # Loop over all elements that need to be sorted one-by-one
    for index_to_be_sorted in range(1, len(array_output)):
        # Move element to the left, until it has been sorted
        for index in range(index_to_be_sorted, 0, -1):
            if array_output[index - 1] <= array_output[index]:
                # Element has been sorted
                break
            # Move element to the left
            _swap_elements(array_output, index, index - 1)

    return array_output


def merge_sort_top_down(array: list[ValueType]) -> list[ValueType]:
    """
    An efficient top-down, divide-and-conquer algorithm.

    Logic
    -----
    Recursively divide input into subarrays, sort them, and merge them back.
    Requires a single buffer array, which gets efficiently copied.

    Complexity
    ----------
    Time: O(N log N)
    Space: O(N log N)

    Comparison
    ----------
    + Stable
    + Same worst- and average-case complexity
    - Not cache friendly
    - More memory

    Notes
    -----
    To derive the time complexity, you write the total processing time as

    T(N) = 2T(N/2) + M(N),

    where M(N) is the time it takes to merge N elements. Using this equation
    recursively, and considering that the total number of recursions is log2 N,
    and that M(N) ~ N, we end up with T(N) ~ N log2 N
    """
    source = list(array).copy()
    target = source.copy()
    _divide_sort_and_merge(source, target, index_left=0, index_right=len(source))
    return target


def _divide_sort_and_merge(source: list[ValueType], target: list[ValueType], index_left: int, index_right: int) -> None:
    if index_right - index_left <= 1:
        # No more splitting possible
        return
    # Split in half
    index_middle = (index_left + index_right) // 2
    # Recursively sort the two halves. Notice how source and target are swapped.
    # This is an optimization: At each merge, the target contains the sorted
    # subarray, which is then used as the source in the next recursion. This
    # back-and-forth merging avoids unnecessary copying.
    _divide_sort_and_merge(target, source, index_left, index_middle)
    _divide_sort_and_merge(target, source, index_middle, index_right)
    _sort_and_merge(source, target, index_left, index_middle, index_right)


def _sort_and_merge(
    source: list[ValueType], target: list[ValueType], index_left: int, index_middle: int, index_right: int
) -> None:
    # Simultaneously merge and sort subarrays [index_left, index_middle) and
    # [index_middle, index_right) from source to target.
    index_sub_left = index_left
    index_sub_right = index_middle
    for index_target in range(index_left, index_right):
        is_left_empty = index_sub_left == index_middle
        is_right_empty = index_sub_right == index_right
        should_merge_from_left = (not is_left_empty) and (
            is_right_empty or source[index_sub_left] <= source[index_sub_right]  # The equality makes it stable
        )

        if should_merge_from_left:
            target[index_target] = source[index_sub_left]
            index_sub_left += 1
        else:
            target[index_target] = source[index_sub_right]
            index_sub_right += 1


def merge_sort_bottom_up(array: list[ValueType]) -> list[ValueType]:
    """
    An efficient bottom-up, divide-and-conquer algorithm.

    Logic
    -----
    Similar to the top-down approach, but without recursion for the division
    step. Instead, it already considers the source array divided into single-
    element arrays, and only performs the merge.

    Complexity
    ----------
    Time: O(N log N)
    Space: O(N)

    Comparison
    ----------
    + Stable
    + Same worst- and average-case complexity
    - Not cache friendly
    - More memory
    """
    source = list(array).copy()
    target = source.copy()
    total_size = len(target)

    # Iteratively merge divided arrays at different scales
    width = 1
    while width < total_size:
        # This swapping is an optimization: At each merge, the target contains
        # the sorted subarray, which is then used as the source in the next pass
        # This back-and-forth merging avoids unnecessary copying.
        source, target = target, source

        # Iteratively sort and merge fixed-width arrays
        for index_left in range(0, total_size, 2 * width):
            index_middle = min(index_left + width, total_size)
            index_right = min(index_left + 2 * width, total_size)
            _sort_and_merge(source, target, index_left, index_middle, index_right)

        width *= 2

    return target


def quicksort_lomuto(array: list[ValueType]) -> list[ValueType]:
    """
    Efficient divide-and-conquer algorithm, based on Lomuto partitioning scheme.

    Logic
    -----
    The algorithm consists of recursively performing the following steps:
    1. Select the last element of the array as pivot.
    2. Partition the array, so that all smaller and larger elements are placed
    on the left and right of the pivot, respectively.

    Complexity
    ----------
    Time: O(N**2)
    Space: O(N)

    Comparison
    ----------
    + Cache friendly
    + Low memory usage
    - Not stable
    - Poor worst-case complexity

    Notes
    -----
    Due to the way elements are re-arranged around the pivot, this is not a
    stable algorithm. The space complexity is linear due to the recursive call
    stack.
    """
    array_output = list(array).copy()
    _quicksort(
        array_output,
        index_left=0,
        index_right=len(array_output),
        partition=_partition_and_return_pivot_index_lomuto,
    )
    return array_output


def quicksort_hoare(array: list[ValueType]) -> list[ValueType]:
    """
    Efficient divide-and-conquer algorithm, based on Hoare partitioning scheme.

    Logic
    -----
    The algorithm consists of recursively performing the following steps:
    1. Select the first element of the array as pivot.
    2. Partition the array, so that all smaller and larger elements are placed
    on the left and right of the pivot, respectively.

    Complexity
    ----------
    Time: O(N**2)
    Space: O(N)

    Comparison
    ----------
    + Cache friendly
    + Low memory usage
    - Not stable
    - Poor worst-case complexity

    Notes
    -----
    Due to the way elements are re-arranged around the pivot, this is not a
    stable algorithm. The space complexity is linear due to the recursive call
    stack.
    """
    array_output = list(array).copy()
    _quicksort(
        array_output,
        index_left=0,
        index_right=len(array_output),
        partition=_partition_and_return_pivot_index_hoare,
    )
    return array_output


_PartitionType = Callable[[list[ValueType], int, int], int]


def _quicksort(array: list[ValueType], index_left: int, index_right: int, partition: _PartitionType) -> None:
    if index_right - index_left <= 1:
        return
    index_pivot = partition(array, index_left, index_right)
    _quicksort(array, index_left, index_pivot, partition)
    _quicksort(array, index_pivot + 1, index_right, partition)


def _partition_and_return_pivot_index_lomuto(array: list[ValueType], index_left: int, index_right: int) -> int:
    """Select last element as pivot, partition array, and return pivot index.

    By default, the pivot point will be placed at the beginning of the array.
    Each time a smaller element is encountered, it is moved to the left of the
    pivot, so the index of the latter gets incremented by one.
    """
    value_pivot = array[index_right - 1]
    index_pivot = index_left
    for index in range(index_left, index_right - 1):
        if array[index] <= value_pivot:
            # Move element to the left of pivot.
            _swap_elements(array, index_pivot, index)
            index_pivot += 1
    # Move pivot after smaller element, and return its index.
    _swap_elements(array, index_pivot, index_right - 1)
    return index_pivot


def _partition_and_return_pivot_index_hoare(array: list[ValueType], index_left: int, index_right: int) -> int:
    """Select first element as pivot, partition array, and return pivot index.

    Successively swap elements between the left and right side of the pivot,
    starting from the ends and moving towards the middle, until they are all
    ordered with respect to the pivot.
    """
    value_pivot = array[index_left]
    index_sub_left = index_left
    index_sub_right = index_right - 1
    while True:
        while array[index_sub_left] < value_pivot:
            # Loop over left subarray, until an unordered element is found
            index_sub_left += 1
        while array[index_sub_right] > value_pivot:
            # Loop over right subarray, until an unordered element is found
            index_sub_right -= 1
        if index_sub_left >= index_sub_right:
            # Partitioning has finished
            return index_sub_right
        # Reorder elements
        _swap_elements(array, index_sub_left, index_sub_right)


def selection_sort(array: list[ValueType]) -> list[ValueType]:
    """
    Simple algorithm constructing sorted array from smallest to largest element.

    Logic
    -----
    It selects the smallest element one by one, and moves them to the end of the
    sorted portion of the array.

    Complexity
    ----------
    Time: O(N**2)
    Space: O(1)
    """
    array_output = list(array).copy()

    # Progressively construct sorted portion
    for size_sorted in range(len(array_output) - 1):
        # Select smallest element in unsorted portion
        index_min = size_sorted
        for index in range(size_sorted + 1, len(array_output)):
            if array_output[index] < array_output[index_min]:
                index_min = index
        # Move to end of sorted portion
        _swap_elements(array_output, size_sorted, index_min)

    return array_output


def _swap_elements(array: list[ValueType], index_1: int, index_2: int) -> None:
    if index_1 != index_2:
        array[index_1], array[index_2] = array[index_2], array[index_1]
