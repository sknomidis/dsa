from __future__ import annotations

import abc
import dataclasses
from typing import TypeVar

ValueType = TypeVar("ValueType")


class Queue(abc.ABC):
    """Fundamental FIFO data structure."""

    @abc.abstractmethod
    def enqueue(self, value: ValueType) -> None:
        """Place value at the back of queue."""

    @abc.abstractmethod
    def dequeue(self) -> ValueType:
        """Pop value in front of queue."""


class ArrayQueue(Queue):
    """Queue data structure with an array implementation.

    Pros
    ----
    - Cache friendly (contiguous storage)

    Cons
    ----
    - Unused memory
    - Resizing is expensive
    - All values need to be of the same type
    """

    def __init__(self, max_size: int = 4) -> None:
        self._array = max_size * [None]
        self._index = 0
        self._size = 0

    @property
    def max_size(self) -> int:
        return len(self._array)

    def enqueue(self, value: ValueType) -> None:
        # O(1)
        assert self._size < self.max_size
        index_last = self._index % self.max_size
        self._array[index_last] = value
        self._index = (self._index + 1) % self.max_size
        self._size += 1

    def dequeue(self) -> ValueType:
        # O(1)
        assert self._size > 0
        index_first = (self._index - self._size + self.max_size) % self.max_size
        value = self._array[index_first]
        self._array[index_first] = None
        self._size -= 1
        return value


class ListQueue(Queue):
    """Queue data structure with a doubly-linked list implementation.

    Pros
    ----
    - Memory efficient
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
        if None in [self._head, self._tail]:
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
            node_enqueued.tail = self._tail
            self._tail.head = node_enqueued
        self._tail = node_enqueued

    def dequeue(self) -> ValueType:
        # O(1)
        assert not self.empty
        node_dequeued = self._head
        self._head = node_dequeued.head
        if self._head is not None:
            self._head.tail = None
        else:
            # List is now empty
            self._tail = None
        return node_dequeued.value


@dataclasses.dataclass()
class _Node:
    value: ValueType
    head: _Node | None = None
    tail: _Node | None = None
