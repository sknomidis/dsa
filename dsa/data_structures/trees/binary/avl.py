from __future__ import annotations

from dsa.data_structures.trees.binary import base


class AVLTree(base.BinarySearchTree):
    """AVL self-balancing tree.

    In an AVL tree, height difference between the left and right branches for
    each node is at most one. This keeps the total height at log(N), which in
    turn limits insertion, deletion, and search complexity to O(log N).

    Pros
    ----
    - simple
    - fast search, due to strict balancing

    Cons
    ----
    - slow insertion and deletion, due to more complex balancing
    """

    @classmethod
    def _insert_value_and_return_root(cls, root: base.BinaryNode | None, value: base.ValueType) -> base.BinaryNode:
        root = super()._insert_value_and_return_root(root, value)
        root = cls._rebalance_subtree_and_return_new_root(root)
        return root

    @classmethod
    def _delete_value_and_return_root(cls, root: base.BinaryNode, value: base.ValueType) -> base.BinaryNode | None:
        root = super()._delete_value_and_return_root(root, value)
        if root is not None:
            root = cls._rebalance_subtree_and_return_new_root(root)
        return root

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

        if balance > 1:
            # Left skewed
            if balance_child_left >= 0:
                # left-left
                return cls._right_rotation(root)
            else:
                # left-right
                root.child_left = cls._left_rotation(root.child_left)
                return cls._right_rotation(root)
        if balance < -1:
            # Right skewed
            if balance_child_right <= 0:
                # right-right
                return cls._left_rotation(root)
            else:
                # right-left
                root.child_right = cls._right_rotation(root.child_right)
                return cls._left_rotation(root)
        return root

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
