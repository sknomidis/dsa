from __future__ import annotations

import abc
import dataclasses
from typing import TypeVar

ValueType = TypeVar("ValueType")


class Stack(abc.ABC):
    """Fundamental LIFO data structure."""

    @abc.abstractmethod
    def push(self, element: ValueType) -> None:
        """Insert value to the top of stack."""

    @abc.abstractmethod
    def pop(self) -> ValueType:
        """Pop value from the top of stack."""


class ArrayStack(Stack):
    """Stack with underlying array implementation.

    Pros
    ----
    - Cache friendly (contiguous storage)

    Cons
    ----
    - Unused memory
    - Resizing is expensive
    - All values need to be of the same type
    """

    def __init__(self, size_max: int = 20) -> None:
        assert size_max > 0
        self._array = size_max * [None]
        self._index_top = 0

    def push(self, element: ValueType) -> None:
        # O(1)
        assert self._index_top < len(self._array), "Max stack size exceeded"
        self._array[self._index_top] = element
        self._index_top += 1

    def pop(self) -> ValueType:
        # O(1)
        assert self._index_top > 0
        self._index_top -= 1
        return self._array[self._index_top]


class ListStack(Stack):
    """Stack with underlying singly-linked list implementation.

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
        self._node_top: _Node | None = None

    def push(self, element: ValueType) -> None:
        # O(1)
        self._node_top = _Node(element, tail=self._node_top)

    def pop(self) -> ValueType:
        # O(1)
        assert self._node_top is not None
        node_popped = self._node_top
        self._node_top = node_popped.tail
        return node_popped.value


@dataclasses.dataclass(frozen=True)
class _Node:
    value: ValueType
    tail: _Node | None
