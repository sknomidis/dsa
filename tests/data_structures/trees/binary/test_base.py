from __future__ import annotations

import pytest

from dsa.data_structures import trees
from dsa.data_structures.trees.binary.base import BinaryNode


@pytest.fixture()
def binary_tree() -> trees.BinaryTree:
    #      2
    #    /   \
    #   1    10
    #  / \   /
    # 4   3 6
    #    /
    #   0
    tree = trees.BinaryTree()
    tree._root = BinaryNode(2)
    tree._root.child_left = BinaryNode(1)
    tree._root.child_left.child_left = BinaryNode(4)
    tree._root.child_left.child_right = BinaryNode(3)
    tree._root.child_left.child_right.child_left = BinaryNode(0)
    tree._root.child_right = BinaryNode(10)
    tree._root.child_right.child_left = BinaryNode(6)
    return tree


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


def test_binary_tree_is_balanced() -> None:
    tree = trees.BinaryTree()
    assert tree.is_balanced()
    root = BinaryNode(0)
    tree._root = root
    assert tree.is_balanced()
    root.child_left = BinaryNode(1)
    assert tree.is_balanced()
    root.child_left.child_left = BinaryNode(2)
    assert not tree.is_balanced()
    root.child_right = BinaryNode(3)
    assert tree.is_balanced()
    root.child_left.child_right = BinaryNode(4)
    assert tree.is_balanced()
    root.child_left.child_right.child_left = BinaryNode(5)
    assert not tree.is_balanced()
    root.child_right.child_right = BinaryNode(6)
    assert tree.is_balanced()


def test_binary_tree_is_complete() -> None:
    binary_tree = trees.BinaryTree()
    assert binary_tree.is_complete()
    root = BinaryNode(0)
    binary_tree._root = root
    assert binary_tree.is_complete()
    binary_tree._root.child_right = BinaryNode(2)
    assert not binary_tree.is_complete()
    binary_tree._root.child_left = BinaryNode(1)
    assert binary_tree.is_complete()
    binary_tree._root.child_left.child_left = BinaryNode(3)
    assert binary_tree.is_complete()
    binary_tree._root.child_right.child_left = BinaryNode(5)
    assert not binary_tree.is_complete()
    binary_tree._root.child_left.child_right = BinaryNode(4)
    assert binary_tree.is_complete()
    binary_tree._root.child_left.child_left.child_right = BinaryNode(5)
    assert not binary_tree.is_complete()


def test_binary_tree_is_degenerate() -> None:
    binary_tree = trees.BinaryTree()
    assert not binary_tree.is_degenerate()
    root = BinaryNode(0)
    binary_tree._root = root
    assert not binary_tree.is_degenerate()
    root.child_left = BinaryNode(1)
    assert binary_tree.is_degenerate()
    root.child_left.child_right = BinaryNode(2)
    assert binary_tree.is_degenerate()
    root.child_left.child_right.child_left = BinaryNode(3)
    assert binary_tree.is_degenerate()
    root.child_left.child_right.child_right = BinaryNode(4)
    assert not binary_tree.is_degenerate()


def test_binary_tree_is_full() -> None:
    binary_tree = trees.BinaryTree()
    for index in range(20):
        binary_tree.insert(index)
        if index % 2 == 0:
            assert binary_tree.is_full()
        else:
            assert not binary_tree.is_full()


def test_binary_tree_is_perfect() -> None:
    binary_tree = trees.BinaryTree()
    assert binary_tree.is_perfect()
    root = BinaryNode(0)
    binary_tree._root = root
    assert binary_tree.is_perfect()
    root.child_left = BinaryNode(1)
    assert not binary_tree.is_perfect()
    root.child_right = BinaryNode(2)
    assert binary_tree.is_perfect()
    root.child_left.child_left = BinaryNode(3)
    assert not binary_tree.is_perfect()
    root.child_left.child_right = BinaryNode(4)
    assert not binary_tree.is_perfect()
    root.child_right.child_left = BinaryNode(5)
    assert not binary_tree.is_perfect()
    root.child_right.child_right = BinaryNode(6)
    assert binary_tree.is_perfect()


def test_binary_tree_insert() -> None:
    #          0
    #        /    \
    #       1      2
    #     /  \    /  \
    #    3     4 5    6
    #   / \   /
    #  7   8 9
    binary_tree = trees.BinaryTree()
    for value in range(10):
        assert not binary_tree.search(value)
        binary_tree.insert(value)
        assert binary_tree.search(value)
    assert binary_tree._root.value == 0
    assert binary_tree._root.child_left.value == 1
    assert binary_tree._root.child_right.value == 2
    assert binary_tree._root.child_left.child_left.value == 3
    assert binary_tree._root.child_left.child_right.value == 4
    assert binary_tree._root.child_right.child_left.value == 5
    assert binary_tree._root.child_right.child_right.value == 6
    assert binary_tree._root.child_left.child_left.child_left.value == 7
    assert binary_tree._root.child_left.child_left.child_right.value == 8
    assert binary_tree._root.child_left.child_right.child_left.value == 9


def test_binary_tree_delete(binary_tree: trees.BinaryTree) -> None:
    for value in [2, 1, 10, 4, 3, 6, 0]:
        assert binary_tree.search(value)
        binary_tree.delete(value)
        assert not binary_tree.search(value)


def test_binary_tree_traversal_BFS(binary_tree: trees.BinaryTree) -> None:
    traversed = list(binary_tree.traversal_BFS())
    assert traversed == [2, 1, 10, 4, 3, 6, 0]


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("preorder", [2, 1, 4, 3, 0, 10, 6]),
        ("inorder", [4, 1, 0, 3, 2, 6, 10]),
        ("postorder", [4, 0, 3, 1, 6, 10, 2]),
    ],
    ids=["preorder", "postorder", "inorder"],
)
def test_binary_tree_traversal_preorder_DFS(variant: str, expected: list[int], binary_tree: trees.BinaryTree) -> None:
    traversed = list(binary_tree.traversal_DFS(variant))
    assert traversed == expected


def test_binary_search_tree_consistency() -> None:
    elements = [42, 43, 10, 11, 8]
    elements_iter = iter(elements)
    binary_tree = trees.BinarySearchTree()
    binary_tree.assert_consistent()
    root = BinaryNode(next(elements_iter))
    binary_tree._root = root
    binary_tree.assert_consistent()
    root.child_right = BinaryNode(next(elements_iter))
    binary_tree.assert_consistent()
    root.child_left = BinaryNode(next(elements_iter))
    binary_tree.assert_consistent()
    root.child_left.child_right = BinaryNode(next(elements_iter))
    binary_tree.assert_consistent()
    root.child_left.child_left = BinaryNode(next(elements_iter))
    binary_tree.assert_consistent()
    assert list(binary_tree.traversal_DFS("inorder")) == sorted(elements)

    root.child_left.child_right.child_left = BinaryNode(12)
    with pytest.raises(AssertionError):
        binary_tree.assert_consistent()


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
    binary_search_tree.assert_consistent()
    assert not binary_search_tree.is_full()
    for value in [35, 2, 65, 6]:
        assert not binary_search_tree.search(value)
        binary_search_tree.insert(value)
        assert binary_search_tree.search(value)
        binary_search_tree.assert_consistent()
    assert binary_search_tree.is_full()


def test_binary_search_tree_delete(binary_search_tree: trees.BinarySearchTree) -> None:
    for value in [28, 42, 4, 23]:
        assert binary_search_tree.search(value)
        binary_search_tree.delete(value)
        assert not binary_search_tree.search(value)
        binary_search_tree.assert_consistent()
    assert list(binary_search_tree.traversal_DFS("inorder")) == [33, 56, 60]

    assert not binary_search_tree.search(200)
    binary_search_tree.delete(200)
    assert not binary_search_tree.search(200)
