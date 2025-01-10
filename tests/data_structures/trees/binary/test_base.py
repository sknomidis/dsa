from __future__ import annotations

import pytest

from dsa.data_structures import trees


@pytest.fixture()
def binary_tree() -> trees.BinaryTree:
    #      2
    #    /   \
    #   1    10
    #  / \   /
    # 4   3 6
    #    /
    #   0
    root = trees.BinaryNode(2)
    root.child_left = trees.BinaryNode(1)
    root.child_left.child_left = trees.BinaryNode(4)
    root.child_left.child_right = trees.BinaryNode(3)
    root.child_left.child_right.child_left = trees.BinaryNode(0)
    root.child_right = trees.BinaryNode(10)
    root.child_right.child_left = trees.BinaryNode(6)
    return trees.BinaryTree(root)


def test_binary_tree_is_balanced() -> None:
    binary_tree = trees.BinaryTree()
    assert binary_tree.is_balanced()
    root = trees.BinaryNode(0)
    binary_tree._root = root
    assert binary_tree.is_balanced()
    root.child_left = trees.BinaryNode(1)
    assert binary_tree.is_balanced()
    root.child_left.child_left = trees.BinaryNode(2)
    assert not binary_tree.is_balanced()
    root.child_right = trees.BinaryNode(3)
    assert binary_tree.is_balanced()
    root.child_left.child_right = trees.BinaryNode(4)
    assert binary_tree.is_balanced()
    root.child_left.child_right.child_left = trees.BinaryNode(5)
    assert not binary_tree.is_balanced()
    root.child_right.child_right = trees.BinaryNode(6)
    assert binary_tree.is_balanced()


def test_binary_tree_is_complete() -> None:
    binary_tree = trees.BinaryTree()
    assert binary_tree.is_complete()
    root = trees.BinaryNode(0)
    binary_tree._root = root
    assert binary_tree.is_complete()
    binary_tree._root.child_right = trees.BinaryNode(2)
    assert not binary_tree.is_complete()
    binary_tree._root.child_left = trees.BinaryNode(1)
    assert binary_tree.is_complete()
    binary_tree._root.child_left.child_left = trees.BinaryNode(3)
    assert binary_tree.is_complete()
    binary_tree._root.child_right.child_left = trees.BinaryNode(5)
    assert not binary_tree.is_complete()
    binary_tree._root.child_left.child_right = trees.BinaryNode(4)
    assert binary_tree.is_complete()
    binary_tree._root.child_left.child_left.child_right = trees.BinaryNode(5)
    assert not binary_tree.is_complete()


def test_binary_tree_is_degenerate() -> None:
    binary_tree = trees.BinaryTree()
    assert not binary_tree.is_degenerate()
    root = trees.BinaryNode(0)
    binary_tree._root = root
    assert not binary_tree.is_degenerate()
    root.child_left = trees.BinaryNode(1)
    assert binary_tree.is_degenerate()
    root.child_left.child_right = trees.BinaryNode(2)
    assert binary_tree.is_degenerate()
    root.child_left.child_right.child_left = trees.BinaryNode(3)
    assert binary_tree.is_degenerate()
    root.child_left.child_right.child_right = trees.BinaryNode(4)
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
    root = trees.BinaryNode(0)
    binary_tree._root = root
    assert binary_tree.is_perfect()
    root.child_left = trees.BinaryNode(1)
    assert not binary_tree.is_perfect()
    root.child_right = trees.BinaryNode(2)
    assert binary_tree.is_perfect()
    root.child_left.child_left = trees.BinaryNode(3)
    assert not binary_tree.is_perfect()
    root.child_left.child_right = trees.BinaryNode(4)
    assert not binary_tree.is_perfect()
    root.child_right.child_left = trees.BinaryNode(5)
    assert not binary_tree.is_perfect()
    root.child_right.child_right = trees.BinaryNode(6)
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
        binary_tree.insert(value)
    assert list(binary_tree.traversal_BFS()) == list(range(10))
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
