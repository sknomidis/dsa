from __future__ import annotations

import pytest

from dsa.data_structures import linked_lists


@pytest.mark.parametrize("values", [[], [1], [1, 2], [1, 2, 4], [1, 2, 4, 8]], ids=lambda l: f"{len(l)}-element(s)")  # noqa
@pytest.mark.parametrize(
    "list_type",
    [
        linked_lists.SinglyLinkedList,
        linked_lists.DoublyLinkedList,
        linked_lists.CircularSinglyLinkedList,
    ],
    ids=lambda c: c.__name__,
)
def test_linked_list_traversal(values: list[int], list_type: type[linked_lists.LinkedList]) -> None:
    linked_list = list_type(values)
    assert list(linked_list) == values


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=lambda i: f"index-{i}")
@pytest.mark.parametrize(
    "list_type",
    [
        linked_lists.SinglyLinkedList,
        linked_lists.DoublyLinkedList,
        linked_lists.CircularSinglyLinkedList,
    ],
    ids=lambda c: c.__name__,
)
def test_linked_list_peek(index: list[int], list_type: type[linked_lists.LinkedList]) -> None:
    for list_size in range(index + 1, 4):
        values = list(range(list_size))[::-1]
        linked_list = list_type(values)
        assert linked_list[index] == values[index]


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=lambda i: f"index-{i}")
@pytest.mark.parametrize(
    "list_type",
    [
        linked_lists.SinglyLinkedList,
        linked_lists.DoublyLinkedList,
        linked_lists.CircularSinglyLinkedList,
    ],
    ids=lambda c: c.__name__,
)
def test_linked_list_insert(index: int, list_type: type[linked_lists.LinkedList]) -> None:
    for list_size in range(index, 4):
        values = list(range(list_size))[::-1]
        linked_list = list_type(values)
        linked_list.insert(index, 42)
        values.insert(index, 42)
        assert list(linked_list) == values


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=lambda i: f"index-{i}")
@pytest.mark.parametrize(
    "list_type",
    [
        linked_lists.SinglyLinkedList,
        linked_lists.DoublyLinkedList,
        linked_lists.CircularSinglyLinkedList,
    ],
    ids=lambda c: c.__name__,
)
def test_linked_list_delete(index: int, list_type: type[linked_lists.LinkedList]) -> None:
    for list_size in range(index + 1, 4):
        values = list(range(list_size))[::-1]
        linked_list = list_type(values)
        linked_list.delete(index)
        values.pop(index)
        assert list(linked_list) == values


def test_doubly_linked_list_inverse_traverse() -> None:
    values = [1, 2, 4, 8]
    linked_list = linked_lists.DoublyLinkedList(values)
    assert list(linked_list.traverse_reverse()) == values[::-1]
