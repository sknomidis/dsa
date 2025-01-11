from __future__ import annotations

import abc
import math
from typing import Iterator, Sequence, TypeVar

ValueType = TypeVar("ValueType")


class Heap(abc.ABC):
    """A complete binary tree used to efficiently retrieve min or max element.

    In the min (max) heap, the minimum (maximum) element is always at the root.
    The value of each node should be smaller (larger) than that of its children.

    An array tree representation is used under the hood, for efficiency.

    Applications
    ------------
    Heaps are typically used for the following:
    - Heap sort
    - Priority queue
    - Graph algorithms
    """

    def __init__(self, elements: Sequence[ValueType] | None = None, max_size: int = 100) -> None:
        self._size = 0
        self._array: list[ValueType | None] = max_size * [None]
        if elements is not None:
            assert len(elements) <= max_size
            for element in elements:
                self.insert(element)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> int:
        return self.size == 0

    def __iter__(self) -> Iterator[ValueType]:
        """Breadth-First Search tree traversal (level order).

        Complexity
        ----------
        Time: O(N)
        Space: O(1)
        """
        for value in self._array[: self.size]:
            yield value

    def get_extremum(self) -> ValueType:
        """Return extremum value from heap.

        Complexity
        ----------
        Time: O(1)
        Space: O(1)
        """
        return self._array[0]

    def insert(self, value: ValueType) -> None:
        """Insert new element to heap.

        Complexity
        ----------
        Time: O(log N)
        Space: O(1)
        """
        assert self.size < len(self._array)

        # Insert element at the end
        index = self._size
        index_parent = self._get_index_parent(index)
        self._array[index] = value
        self._size += 1

        # Move element up until heap property is satisfied
        while index > 0 and not self._is_heap(index_parent, index):
            self._swap_elements(index, index_parent)
            index = index_parent
            index_parent = self._get_index_parent(index)

    def extract_extremum(self) -> ValueType:
        """Remove and return extremum.

        Complexity
        ----------
        Time: O(log N)
        Space: O(log N)
        """
        assert self.size > 0
        # Extract extremum
        value = self._array[0]
        # Move last element to root and update size
        self._array[0] = self._array[self.size - 1]
        self._size -= 1
        if self.size == 1:
            return value
        # Restore heap property
        self._heapify(index_root=0)
        return value

    def delete(self, value: ValueType) -> None:
        """Remove element from heap.

        Complexity
        ----------
        Time: O(N), searching is the bottleneck
        Space: O(log N)
        """
        assert self.size > 0
        # Find element and its parent
        index = self._find_element(value)
        index_parent = self._get_index_parent(index)
        # Make element the new extremum
        self._array[index] = self._get_most_extreme_possible_value()
        # Move element up until heapify property is restored.
        while index > 0 and not self._is_heap(index_parent, index):
            self._swap_elements(index, self._get_index_parent(index))
            index = self._get_index_parent(index)
            index_parent = self._get_index_parent(index)
        # Extract artificially-extreme value from heap
        self.extract_extremum()

    def _heapify(self, index_root: int) -> None:
        """Progressively move root down until heap property is satisfied.

        Complexity
        ----------
        Time: O(log N)
        Space: O(log N)
        """
        index_left = self._get_index_child_left(index_root)
        index_right = self._get_index_child_right(index_root)
        index_extremum = index_root
        if index_left < self.size and not self._is_heap(index_root, index_left):
            # Left child should be moved up
            index_extremum = index_left
        if index_right < self.size and not self._is_heap(index_extremum, index_right):
            # Right child should be moved up
            index_extremum = index_right
        if index_extremum != index_root:
            # Move child up and repeat in branch
            self._swap_elements(index_root, index_extremum)
            self._heapify(index_extremum)

    def _find_element(self, value: ValueType) -> int:
        """Search for given element.

        Complexity
        ----------
        Time: O(N)
        Space: O(1)
        """
        for index, value_stored in enumerate(self._array[: self.size]):
            if value_stored == value:
                return index
        raise AssertionError("Value not found")

    def _swap_elements(self, index_1: int, index_2: int) -> None:
        self._array[index_1], self._array[index_2] = self._array[index_2], self._array[index_1]

    @staticmethod
    def _get_index_parent(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def _get_index_child_left(index: int) -> int:
        return 2 * index + 1

    @staticmethod
    def _get_index_child_right(index: int) -> int:
        return 2 * index + 2

    @abc.abstractmethod
    def _is_heap(self, index_parent: ValueType, index_child: ValueType) -> bool: ...

    @staticmethod
    @abc.abstractmethod
    def _get_most_extreme_possible_value() -> ValueType: ...


class MinHeap(Heap):
    """The value of each node is smaller than that of its children."""

    def _is_heap(self, index_parent: ValueType, index_child: ValueType) -> bool:
        return self._array[index_parent] < self._array[index_child]

    @staticmethod
    def _get_most_extreme_possible_value() -> ValueType:
        return -math.inf


class MaxHeap(Heap):
    """The value of each node is larger than that of its children."""

    def _is_heap(self, index_parent: ValueType, index_child: ValueType) -> bool:
        return self._array[index_parent] > self._array[index_child]

    @staticmethod
    def _get_most_extreme_possible_value() -> ValueType:
        return math.inf
