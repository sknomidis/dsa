from __future__ import annotations

import abc
import dataclasses
from typing import TypeVar

ValueType = TypeVar("ValueType")


class Queue(abc.ABC):
    """Fundamental FIFO data structure."""

    @abc.abstractmethod
    def enqueue(self, value: ValueType) -> None:
        """Insert value to the back of queue."""

    @abc.abstractmethod
    def dequeue(self) -> ValueType:
        """Pop value from front of queue."""


class ArrayQueue(Queue):
    """Queue with underlying array implementation.

    Pros
    ----
    - Cache friendly (contiguous storage)

    Cons
    ----
    - Unused memory
    - Resizing is expensive
    - All values need to be of the same type
    """

    def __init__(self, size_max: int = 4) -> None:
        self._array = size_max * [None]
        self._index_tail = 0
        self._size = 0

    @property
    def size_max(self) -> int:
        return len(self._array)

    def enqueue(self, value: ValueType) -> None:
        # O(1)
        assert self._size < self.size_max, "Max queue size exceeded"
        index_tail = self._index_tail % self.size_max
        self._array[index_tail] = value
        self._index_tail = (self._index_tail + 1) % self.size_max
        self._size += 1

    def dequeue(self) -> ValueType:
        # O(1)
        assert self._size > 0
        index_head = (self._index_tail - self._size + self.size_max) % self.size_max
        value = self._array[index_head]
        self._array[index_head] = None
        self._size -= 1
        return value


class ListQueue(Queue):
    """Queue with underlying doubly-linked list implementation.

    Pros
    ----
    - Memory efficient
    - Cheap resizing
    - Values do not have to be of same type

    Cons
    ----
    - Not cache friendly
    """

    def __init__(self) -> None:
        self._head: _Node | None = None
        self._tail: _Node | None = None

    @property
    def empty(self) -> bool:
        if self._head is None or self._tail is None:
            assert self._head is self._tail is None
            return True
        return False

    def enqueue(self, value: ValueType) -> None:
        # O(1)
        node_enqueued = _Node(value)
        if self.empty:
            # It will be both the head and tail node
            self._head = node_enqueued
        else:
            # Re-arrange node connections
            node_enqueued.head = self._tail
            self._tail.tail = node_enqueued
        self._tail = node_enqueued

    def dequeue(self) -> ValueType:
        # O(1)
        assert not self.empty
        node_dequeued = self._head
        self._head = node_dequeued.tail
        if self._head is not None:
            self._head.head = None
        else:
            # List is now empty
            self._tail = None
        return node_dequeued.value


@dataclasses.dataclass()
class _Node:
    value: ValueType
    head: _Node | None = None
    tail: _Node | None = None
