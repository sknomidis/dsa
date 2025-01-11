from __future__ import annotations

import pytest

from dsa.data_structures import trees
from dsa.data_structures.trees.binary.base import BinaryNode


@pytest.fixture()
def binary_search_tree() -> trees.BinarySearchTree:
    #     42
    #    /   \
    #   23   60
    #  / \   /
    # 4  33 56
    #    /
    #   28
    tree = trees.BinarySearchTree()
    tree._root = BinaryNode(42)
    tree._root.child_left = BinaryNode(23)
    tree._root.child_left.child_left = BinaryNode(4)
    tree._root.child_left.child_right = BinaryNode(33)
    tree._root.child_left.child_right.child_left = BinaryNode(28)
    tree._root.child_right = BinaryNode(60)
    tree._root.child_right.child_left = BinaryNode(56)
    return tree


def test_binary_search_tree_is_ordered() -> None:
    elements = [42, 43, 10, 11, 8]
    elements_iter = iter(elements)
    binary_tree = trees.BinarySearchTree()
    assert binary_tree.is_ordered()
    root = BinaryNode(next(elements_iter))
    binary_tree._root = root
    assert binary_tree.is_ordered()
    root.child_right = BinaryNode(next(elements_iter))
    assert binary_tree.is_ordered()
    root.child_left = BinaryNode(next(elements_iter))
    assert binary_tree.is_ordered()
    root.child_left.child_right = BinaryNode(next(elements_iter))
    assert binary_tree.is_ordered()
    root.child_left.child_left = BinaryNode(next(elements_iter))
    assert binary_tree.is_ordered()
    assert list(binary_tree.traversal_DFS("inorder")) == sorted(elements)

    root.child_left.child_right.child_left = BinaryNode(12)
    assert not binary_tree.is_ordered()


def test_binary_search_tree_min(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.find_min() == 4
    for value in range(5, 10):
        binary_search_tree.insert(value)
        assert binary_search_tree.find_min() == 4
    binary_search_tree.insert(2)
    assert binary_search_tree.find_min() == 2


def test_binary_search_tree_max(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.find_max() == 60
    for value in range(57, 60):
        binary_search_tree.insert(value)
        assert binary_search_tree.find_max() == 60
    binary_search_tree.insert(240)
    assert binary_search_tree.find_max() == 240


def test_binary_search_tree_floor(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.find_floor(43) == 42
    assert binary_search_tree.find_floor(32) == 23
    assert binary_search_tree.find_floor(12) == 4
    assert binary_search_tree.find_floor(60) == 60
    assert binary_search_tree.find_floor(2) is None


def test_binary_search_tree_ceil(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.find_ceil(40) == 42
    assert binary_search_tree.find_ceil(16) == 23
    assert binary_search_tree.find_ceil(1) == 4
    assert binary_search_tree.find_ceil(60) == 60
    assert binary_search_tree.find_ceil(62) is None


def test_binary_search_tree_insert(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.is_ordered()
    assert not binary_search_tree.is_full()
    binary_search_tree.insert(35)
    assert binary_search_tree.is_ordered()
    binary_search_tree.insert(2)
    assert binary_search_tree.is_ordered()
    binary_search_tree.insert(65)
    assert binary_search_tree.is_ordered()
    binary_search_tree.insert(6)
    assert binary_search_tree.is_ordered()
    assert binary_search_tree.is_full()
    with pytest.raises(AssertionError):
        binary_search_tree.insert(35)


def test_binary_search_tree_delete(binary_search_tree: trees.BinarySearchTree) -> None:
    assert binary_search_tree.is_ordered()
    binary_search_tree.delete(28)
    assert binary_search_tree.is_ordered()
    assert list(binary_search_tree.traversal_BFS()) == [42, 23, 60, 4, 33, 56]
    binary_search_tree.delete(42)
    assert binary_search_tree.is_ordered()
    assert list(binary_search_tree.traversal_BFS()) == [56, 23, 60, 4, 33]
    binary_search_tree.delete(4)
    assert binary_search_tree.is_ordered()
    assert list(binary_search_tree.traversal_BFS()) == [56, 23, 60, 33]
    binary_search_tree.delete(23)
    assert binary_search_tree.is_ordered()
    assert list(binary_search_tree.traversal_BFS()) == [56, 33, 60]
    binary_search_tree.delete(200)
    assert list(binary_search_tree.traversal_BFS()) == [56, 33, 60]
