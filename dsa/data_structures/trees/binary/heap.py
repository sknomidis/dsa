from __future__ import annotations

import abc
import math
from typing import Iterator, TypeVar

ValueType = TypeVar("ValueType")


class Heap(abc.ABC):
    """A complete binary tree used to efficiently retrieve min or max element.

    In the min (max) heap, the minimum (maximum) element is always at the root.
    The value of each node should be smaller (larger) than that of its children.

    Since heap is a complete binary tree, an array representation is used under
    the hood, for efficiency.

    Applications
    ------------
    Heaps are typically used for the following:
    - Heap sort
    - Priority queue
    - Graph algorithms
    """

    def __init__(self, *, size_max: int = 100) -> None:
        self._array: list[ValueType | None] = size_max * [None]
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def size_max(self) -> int:
        return len(self._array)

    def __iter__(self) -> Iterator[ValueType]:
        """Breadth-First Search tree traversal (level order).

        Complexity
        ----------
        Time: O(N)
        Space: O(1)
        """
        yield from self._array[: self.size]

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

        More specifically, it places the new value at the end of the array, and
        progressively moves it up until the heap property is restored.

        Complexity
        ----------
        Time: O(log N)
        Space: O(1)
        """
        assert self.size < self.size_max
        index_inserted = self._size
        self._array[index_inserted] = value
        self._size += 1
        self._heapify_up(index_inserted)

    def pop(self) -> ValueType:
        """Remove and return extremum.

        It replaces the root node with the last one, and progressively moves it
        down until the heap property is restored.

        Complexity
        ----------
        Time: O(log N)
        Space: O(1)
        """
        assert self.size > 0
        value = self._array[0]
        self._array[0] = None
        self._swap_values(0, self.size - 1)
        self._size -= 1
        self._heapify_down(0)
        return value

    def delete(self, value: ValueType) -> None:
        """Remove element from heap.

        It finds the node to remove, replaces its value with the most extreme
        one possible (plus or minus infinity), moves it to the top, and pops it.

        Complexity
        ----------
        Time: O(N)
        Space: O(1)
        """
        assert self.size > 0
        index_deleted = self._array.index(value)
        self._array[index_deleted] = self._get_most_extreme_possible_value()
        self._heapify_up(index_deleted)
        self.pop()

    def _heapify_up(self, index_inserted: int) -> None:
        """Move node up until heap property is satisfied.

        Complexity
        ----------
        Time: O(log N)
        Space: O(1)
        """
        index_parent = self._get_index_parent(index_inserted)
        while index_inserted > 0 and not self._is_heap(index_parent, index_inserted):
            self._swap_values(index_inserted, index_parent)
            index_inserted = index_parent
            index_parent = self._get_index_parent(index_inserted)

    def _heapify_down(self, index_inserted: int) -> None:
        """Move node down until heap property is satisfied.

        Complexity
        ----------
        Time: O(log N)
        Space: O(1)
        """
        while index_inserted < self.size - 1:
            # Find next node with which to swap inserted one
            index_next = index_inserted
            index_left = self._get_index_child_left(index_inserted)
            index_right = self._get_index_child_right(index_inserted)
            if index_left < self.size and not self._is_heap(index_inserted, index_left):
                index_next = index_left
            if index_right < self.size and not self._is_heap(index_next, index_right):
                index_next = index_right

            if index_next != index_inserted:
                # Move child up and repeat in branch
                self._swap_values(index_inserted, index_next)
                index_inserted = index_next
            else:
                # Heap property has been restored
                return

    def _swap_values(self, index_1: int, index_2: int) -> None:
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
