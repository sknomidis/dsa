from __future__ import annotations

import abc
import dataclasses
from typing import Iterator, Sequence, TypeVar

ValueType = TypeVar("ValueType")


class LinkedList(abc.ABC):
    """Linear collection of elements allowing efficient insertion and deletion.

    Pros
    ----
    - Fast insertion/deletion
    - Memory efficient for non-fixed data size
    - Allows for varying element types

    Cons
    ----
    - Slow access
    - More memory needed per element
    - Cache inefficient
    """

    @abc.abstractmethod
    def __init__(self, values: Sequence[ValueType] | None = None) -> None: ...

    def __len__(self) -> int:
        # O(N)
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[ValueType]:
        # O(N)
        for node in self._iterate_nodes():
            yield node.value

    def __getitem__(self, index: int) -> ValueType:
        # O(N)
        return self._get_node(index).value

    def __setitem__(self, index: int, value: ValueType) -> None:
        # O(N)
        self._get_node(index).value = value

    @abc.abstractmethod
    def insert(self, index: int, value: ValueType) -> None: ...

    @abc.abstractmethod
    def delete(self, index: int) -> None: ...

    @abc.abstractmethod
    def _iterate_nodes(self) -> Iterator[_Node]: ...

    def _get_node(self, index: int) -> _Node:
        for index_current, node in enumerate(self._iterate_nodes()):
            if index_current == index:
                return node
        raise IndexError


class SinglyLinkedList(LinkedList):
    """Each node holds a pointer to the next one in the sequence."""

    def __init__(self, values: Sequence[ValueType] | None = None) -> None:
        # O(N)
        self._head = self._create_nodes_and_return_head(values) if values else None

    def insert(self, index: int, value: ValueType) -> None:
        # O(N), since we are not given the previous node, but an index
        node_inserted = _Node(value)
        if index == 0:
            # Place node at beginning of list
            node_inserted.head = self._head
            self._head = node_inserted
        else:
            # Place node between index - 1 and index
            node_tail = self._get_node(index - 1)
            node_inserted.head = node_tail.head
            node_tail.head = node_inserted

    def delete(self, index: int) -> None:
        # O(N), since we are not given the previous node, but an index
        if index == 0:
            self._head = self._head.head
        else:
            node_tail = self._get_node(index - 1)
            node_tail.head = node_tail.head.head

    def _iterate_nodes(self) -> Iterator[_Node]:
        node = self._head
        while node is not None:
            yield node
            node = node.head

    @staticmethod
    def _create_nodes_and_return_head(values: Sequence[ValueType]) -> _Node:
        head = _Node(values[0])
        previous = head
        for value in values[1:]:
            previous.head = _Node(value)
            previous = previous.head
        return head


class DoublyLinkedList(LinkedList):
    """Each node holds a pointer to the previous and next one in the sequence."""

    def __init__(self, values: Sequence[ValueType] | None = None) -> None:
        # O(N)
        self._head, self._tail = self._create_nodes_and_return_head_and_tail(values) if values else (None, None)

    def traverse_reverse(self) -> Iterator[ValueType]:
        node = self._tail
        while node is not None:
            yield node.value
            node = node.tail

    def insert(self, index: int, value: ValueType) -> None:
        # O(N), since we are not given a node, but an index
        node_inserted = _Node(value)
        if len(self) == 0:
            # Only element in list
            assert index == 0
            self._head = node_inserted
            self._tail = node_inserted

        elif index == 0:
            # Insert at the beginning
            node_inserted.head = self._head
            self._head.tail = node_inserted
            self._head = node_inserted

        elif index == len(self):
            # Insert at the end
            node_inserted.tail = self._tail
            self._tail.head = node_inserted
            self._tail = node_inserted

        else:
            # Insert between index - 1 and index
            node_inserted.tail = self._get_node(index - 1)
            node_inserted.head = self._get_node(index)
            node_inserted.tail.head = node_inserted
            node_inserted.head.tail = node_inserted

    def delete(self, index: int) -> None:
        # O(N), since we are not given a node, but an index
        if len(self) == 1:
            # Only element in the list
            assert index == 0
            self._head = None
            self._tail = None

        elif index == 0:
            # Delete first element
            self._head = self._head.head
            self._head.tail = None

        elif index == len(self) - 1:
            # Delete last element
            self._tail = self._tail.tail
            self._tail.head = None

        else:
            # Delete element in middle
            tail = self._get_node(index - 1)
            head = self._get_node(index + 1)
            tail.head = head
            head.tail = tail

    def _iterate_nodes(self) -> Iterator[_Node]:
        node = self._head
        while node is not None:
            yield node
            node = node.head

    @staticmethod
    def _create_nodes_and_return_head_and_tail(values: Sequence[ValueType]) -> tuple[_Node, _Node]:
        head = _Node(values[0])
        previous = head
        for value in values[1:]:
            previous.head = _Node(value)
            previous.head.tail = previous
            previous = previous.head
        tail = previous
        return head, tail


class CircularSinglyLinkedList(LinkedList):
    """Each node holds a pointer to the next, and the last one to the first one."""

    def __init__(self, values: Sequence[ValueType] | None = None) -> None:
        # O(N)
        self._tail = self._create_nodes_and_return_tail(values) if values else None

    def insert(self, index: int, value: ValueType) -> None:
        # O(N), since we are not given a node, but an index
        node_inserted = _Node(value)
        if len(self) == 0:
            # Insert to empty list
            assert index == 0
            node_inserted.head = node_inserted
            self._tail = node_inserted
        elif index == 0:
            # Insert at the beginning
            node_inserted.head = self._tail.head
            self._tail.head = node_inserted
        elif index == len(self):
            # Insert at the end
            node_inserted.head = self._tail.head
            self._tail.head = node_inserted
            self._tail = node_inserted
        else:
            # Insert in the middle
            tail = self._get_node(index - 1)
            node_inserted.head = tail.head
            tail.head = node_inserted

    def delete(self, index: int) -> None:
        # O(N), since we are not given a node, but an index
        assert 0 <= index < len(self)
        node_previous = self._get_previous_node(index)
        if node_previous.head is node_previous:
            self._tail = None
        else:
            node_previous.head = node_previous.head.head

    def _iterate_nodes(self) -> Iterator[_Node]:
        if self._tail is None:
            return
        head_node = self._tail.head
        yield head_node
        node = head_node.head
        while node is not head_node:
            yield node
            node = node.head

    def _get_previous_node(self, index) -> _Node:
        if index == 0:
            return self._tail
        try:
            return self._get_node(index - 1)
        except IndexError:
            return None

    @staticmethod
    def _create_nodes_and_return_tail(values: Sequence[ValueType]) -> _Node:
        head = _Node(values[0])
        previous = head
        for value in values[1:]:
            previous.head = _Node(value)
            previous = previous.head
        tail = previous
        tail.head = head
        return tail


@dataclasses.dataclass()
class _Node:
    value: ValueType
    head: _Node | None = None
    tail: _Node | None = None
