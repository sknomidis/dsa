from __future__ import annotations

from dsa.data_structures.trees.binary import base


class AVLTree(base.BinarySearchTree):
    """AVL self-balancing tree.

    In an AVL tree, height difference between the left and right branches for
    each node is at most one. This keeps the total height at log(N), which in
    turn limits insert, delete, and search complexity to O(log N).

    Comparison with other self-balancing trees, they:
    + are simpler
    + have faster search, due to strict balancing
    - have slower insert and delete, due to more complex balancing
    """

    def insert(self, value: base.ValueType) -> base.ValueType:
        """Insert a new value to AVL tree and rebalance subtrees.

        Note that, there is no support for duplicate values.

        Complexity
        ----------
        Time: O(height)
        Space: O(height)

        where height is log(N).
        """

        def insert_rebalance_and_return_new_root(root: base.BinaryNode | None) -> base.BinaryNode:
            if root is None:
                # Insert value to leaf
                return base.BinaryNode(value)

            # Recursively apply to appropriate subtree
            assert value != root.value, "Duplicates are not allowed"
            if value < root.value:
                root.child_left = insert_rebalance_and_return_new_root(root.child_left)
            else:
                root.child_right = insert_rebalance_and_return_new_root(root.child_right)

            # Rebalance if necessary
            return self._rebalance_subtree_and_return_new_root(root)

        self._root = insert_rebalance_and_return_new_root(self._root)

    def delete(self, value: base.ValueType) -> None:
        """Delete a value from AVL tree and rebalance subtrees.

        An exception is raised if the value is not found.

        Complexity
        ----------
        Time: O(height)
        Space: O(height)

        where height is log(N).
        """

        def delete_rebalance_and_return_new_root(
            root: base.BinaryNode, value: base.ValueType
        ) -> base.BinaryNode | None:
            assert root is not None, "Value not found"

            if value < root.value:
                root.child_left = delete_rebalance_and_return_new_root(root.child_left, value)
            elif value > root.value:
                root.child_right = delete_rebalance_and_return_new_root(root.child_right, value)
            else:
                # Delete node
                if root.num_children == 0:
                    # No children, no worries
                    return None
                if root.num_children == 1:
                    # Only child is moved one level up
                    root = root.child_left or root.child_right
                else:
                    # Replace deleted node with smallest element in right branch
                    root_new = root.child_right
                    while root_new.child_left is not None:
                        root_new = root_new.child_left
                    root.value = root_new.value
                    root.child_right = delete_rebalance_and_return_new_root(root.child_right, root_new.value)

            return self._rebalance_subtree_and_return_new_root(root)

        self._root = delete_rebalance_and_return_new_root(self._root, value)

    @classmethod
    def _rebalance_subtree_and_return_new_root(cls, root: base.BinaryNode) -> base.BinaryNode:
        r"""Rebalance subtree and return updated root.

        We distinguish between the following cases:

        - left-left:

                 z                                      y
                / \                                   /   \
               y   T4      Right Rotate (z)          x      z
              / \          - - - - - - - - ->      /  \    /  \
             x   T3                               T1  T2  T3  T4
            / \
          T1   T2

        - right-right:

            z                                           y
           /  \                                       /   \
          T1   y           Left Rotate(z)            z      x
              /  \         - - - - - - - - ->       / \    / \
             T2   x                                T1  T2 T3  T4
                 / \
               T3  T4

        - left-right:

               z                                          z                                     x
              / \                                       /   \                                  /  \
             y   T4        Left Rotate (y)             x    T4       Right Rotate(z)         y      z
            / \            - - - - - - - - ->         /  \           - - - - - - - ->       / \    / \
          T1   x                                     y    T3                              T1  T2 T3  T4
              / \                                   / \
            T2   T3                               T1   T2

        - right-left:

             z                                       z                                          x
            / \                                     / \                                        /  \
          T1   y           Right Rotate (y)       T1   x              Left Rotate(z)         z      y
              / \          - - - - - - - - ->        /  \           - - - - - - - ->        / \    / \
             x   T4                                 T2   y                                T1  T2  T3  T4
            / \                                         /  \
          T2   T3                                      T3   T4
        """
        # TODO: Store and update node height instead of recomputing.
        balance = root.balance
        balance_child_left = 0 if root.child_left is None else root.child_left.balance
        balance_child_right = 0 if root.child_right is None else root.child_right.balance

        if balance > 1 and balance_child_left >= 0:
            # left-left
            return cls._right_rotation(root)
        if balance < -1 and balance_child_right <= 0:
            # right-right
            return cls._left_rotation(root)
        if balance > 1 and balance_child_left < 0:
            # left-right
            root.child_left = cls._left_rotation(root.child_left)
            return cls._right_rotation(root)
        if balance < -1 and balance_child_right > 0:
            # right-left
            root.child_right = cls._right_rotation(root.child_right)
            return cls._left_rotation(root)
        return root
