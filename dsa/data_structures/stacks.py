from __future__ import annotations

import abc
import dataclasses
from typing import Any


class Stack(abc.ABC):
    @abc.abstractmethod
    def push(self, element: Any) -> None: ...

    @abc.abstractmethod
    def pop(self) -> Any: ...


class ArrayStack(Stack):
    def __init__(self, max_size: int = 20) -> None:
        assert max_size > 0
        self._array = max_size * [None]
        self._index = 0

    def push(self, element: Any) -> None:
        self._array[self._index] = element
        self._index += 1

    def pop(self) -> Any:
        assert self._index > 0
        self._index -= 1
        return self._array[self._index]


class ListStack(Stack):
    def __init__(self) -> None:
        self._node_last: _Node | None = None

    def push(self, element: Any) -> None:
        node_last = self._node_last
        self._node_last = _Node(element, next=node_last)

    def pop(self) -> Any:
        assert self._node_last is not None
        node_popped = self._node_last
        self._node_last = node_popped.next
        return node_popped.value


@dataclasses.dataclass(frozen=True)
class _Node:
    value: Any
    next: _Node | None
