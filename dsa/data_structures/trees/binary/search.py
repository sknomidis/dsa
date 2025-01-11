from __future__ import annotations

from typing import Any

from dsa.data_structures.trees.binary import base


class BinarySearchTree(base.BinaryTree):
    """Binary Search Tree (BST).

    In a BST, the left (right) child of each node contains values smaller
    (greater) than the parent. This allows for efficient searching,
    insertion, and deletion.
    """

    def is_ordered(self) -> bool:
        """Check if it is a BST."""
        for node in self._traversal_BFS():
            if node.child_left is not None and node.child_left.value >= node.value:
                return False
            if node.child_right is not None and node.child_right.value <= node.value:
                return False
        return True

    def find_min(self) -> Any:
        """Find bottom-left-most leaf."""
        root = self._root
        while root.child_left is not None:
            root = root.child_left
        return root.value

    def find_max(self) -> Any:
        """Find bottom-right-most leaf."""
        root = self._root
        while root.child_right is not None:
            root = root.child_right
        return root.value

    def find_floor(self, value: Any) -> Any | None:
        """Find largest value that is smaller than or equal to input.

        Complexity
        ----------
        Time: O(height)
        Space: O(height)
        """

        def find_floor_subtree(root: base.BinaryNode | None) -> Any | None:
            if root is None:
                # No floor exists
                return None
            if root.value > value:
                # Move to left branch
                return find_floor_subtree(root.child_left)
            if root.value == value or root.child_right is None or root.child_right.value > value:
                # Floor has been found
                return root.value
            return find_floor_subtree(root.child_right)

        return find_floor_subtree(self._root)

    def find_ceil(self, value: Any) -> Any:
        """Find smallest value that is larger than or equal to input.

        Complexity
        ----------
        Time: O(height)
        Space: O(height)
        """

        def find_ceil_subtree(root: base.BinaryNode | None) -> Any | None:
            if root is None:
                return None
            if root.value < value:
                return find_ceil_subtree(root.child_right)
            if root.value == value or root.child_left is None or root.child_left.value < value:
                return root.value
            return find_ceil_subtree(root.child_left)

        return find_ceil_subtree(self._root)

    def insert(self, value: Any) -> None:
        """Insert a new value to a BST.

        Note that, there is no support for duplicate values.

        Complexity
        ----------
        Time: O(height)
        Space: O(1)
        """
        node = base.BinaryNode(value)
        if self._root is None:
            self._root = node
            return

        def insert_subtree(root: base.BinaryNode) -> None:
            assert value != root.value, "Duplicate element encountered"
            if value < root.value:
                if root.child_left is None:
                    root.child_left = node
                else:
                    insert_subtree(root.child_left)
            else:
                if root.child_right is None:
                    root.child_right = node
                else:
                    insert_subtree(root.child_right)

        insert_subtree(self._root)

    def delete(self, value: Any) -> None:
        """Delete value from a BST using recursion.

        Recursively search for node to be deleted. Single-child nodes are
        replaced by their child. Full nodes are replaced by the minimum element
        on their right subtree.

        Complexity
        ----------
        Time: O(height)
        Space: O(height)
        """

        def delete_node_from_subtree_and_return_new_root(
            root: base.BinaryNode | None, value: Any
        ) -> base.BinaryNode | None:
            if root is None:
                return None

            # Keep searching for node to delete
            if value < root.value:
                # Move further down left subtree
                root.child_left = delete_node_from_subtree_and_return_new_root(root.child_left, value)
                return root
            if value > root.value:
                # Move further down right subtree
                root.child_right = delete_node_from_subtree_and_return_new_root(root.child_right, value)
                return root

            # Single-child root
            if root.child_left is None:
                # Move right child one level up
                return root.child_right
            if root.child_right is None:
                # Move left child one level up
                return root.child_left

            # Successor will be the smallest element of right subtree
            successor = root.child_right
            while successor.child_left is not None:
                successor = successor.child_left

            # Replace root with successor
            root.value = successor.value
            root.child_right = delete_node_from_subtree_and_return_new_root(root.child_right, successor.value)
            return root

        self._root = delete_node_from_subtree_and_return_new_root(self._root, value)

    @staticmethod
    def _left_rotation(root: base.BinaryNode) -> base.BinaryNode:
        r"""Rotate subtree as follows:
             x              y
            / \            / \
           T1  y    ->    x   T3
              / \        / \
            T2   T3     T1  T2
        """
        x = root
        y = x.child_right
        assert y is not None
        t2 = y.child_left
        x.child_right = t2
        y.child_left = x
        return y

    @staticmethod
    def _right_rotation(root: base.BinaryNode) -> base.BinaryNode:
        r"""Rotate subtree as follows:
             y              x
            / \            / \
           x   T3   ->   T1   y
          / \                / \
        T1   T2            T2   T3
        """
        y = root
        x = root.child_left
        assert x is not None
        t2 = x.child_right
        x.child_right = y
        y.child_left = t2
        return x
