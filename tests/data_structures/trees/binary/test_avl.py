from __future__ import annotations

import numpy as np
import pytest

from dsa.data_structures import trees


def test_AVL_tree_insert(rng: np.random.Generator) -> None:
    elements = list(range(42))
    tree = trees.AVLTree()

    for element in rng.permuted(elements).tolist():
        tree.insert(element)

        assert tree.is_balanced()
    assert list(tree.traversal_DFS("inorder")) == elements


def test_AVL_tree_delete() -> None:
    elements = list(range(42))
    tree = trees.AVLTree()
    for element in elements:
        tree.insert(element)

    for element in elements[1::2]:
        tree.delete(element)

        assert tree.is_balanced()
    assert list(tree.traversal_DFS("inorder")) == elements[::2]
    with pytest.raises(AssertionError):
        tree.delete(100)
