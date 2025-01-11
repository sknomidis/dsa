from __future__ import annotations

import collections
import dataclasses
from typing import Iterator, Literal, Self, TypeVar

ValueType = TypeVar("ValueType")


class BinaryTree:
    """Hierarchical data structure with each node having at most two children.

    It is mainly used for efficient data storage and retrieval.

    There are two main representation methods:
    - Nodes and References: Requires more memory per node and is not cache
      friendly, but it is more flexible and, thus, common.
    - Array: Better locality of reference, but it can be very inefficient for
      non-complete trees. It is typically used in heaps.
    """

    def __init__(self) -> None:
        self._root: _BinaryNode | None = None

    def is_balanced(self) -> bool:
        """Height difference between all left and right subtrees is at most 1."""
        return all(abs(node.balance) <= 1 for node in self._traversal_BFS())

    def is_complete(self) -> bool:
        """All intermediate levels are filled and all leaf nodes are on the left."""
        nonfull_node_has_been_found = False
        for node in self._traversal_BFS():
            if nonfull_node_has_been_found:
                # All subsequent children must be leaf nodes
                if node.num_children > 0:
                    return False
            elif node.num_children < 2:
                nonfull_node_has_been_found = True
                if node.num_children == 1 and node.child_right is not None:
                    return False
        return True

    def is_degenerate(self) -> bool:
        """Each node has 1 child."""
        if self._root is None or self._root.num_children != 1:
            return False
        return all(node.num_children < 2 for node in self._traversal_BFS())

    def is_full(self) -> bool:
        """Each node has either 0 or 2 children."""
        return all(node.num_children % 2 == 0 for node in self._traversal_BFS())

    def is_perfect(self) -> bool:
        """Internal nodes have 2 children, and leaf nodes are on same level."""

        def is_subtree_perfect(root: _BinaryNode, height: int) -> bool:
            if root is None:
                return True
            if root.num_children == 1:
                return False
            if root.num_children == 0:
                # Leaf nodes can only exist at last level
                return height == 1
            is_subtree_perfect_left = is_subtree_perfect(root.child_left, height - 1)
            is_subtree_perfect_right = is_subtree_perfect(root.child_right, height - 1)
            return is_subtree_perfect_left and is_subtree_perfect_right

        height_total = self._root.height if self._root is not None else 0
        return is_subtree_perfect(self._root, height_total)

    def search(self, value: ValueType) -> bool:
        """Return `True` if given value exists in tree, `False` otherwise.

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """
        for node in self._traversal_BFS():
            if node.value == value:
                return True
        return False

    def insert(self, value: ValueType) -> None:
        """Breadth-First Search tree insertion (level order).

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """
        node_inserted = _BinaryNode(value)
        if self._root is None:
            self._root = node_inserted
            return
        for node in self._traversal_BFS():
            if node.child_left is None:
                node.child_left = node_inserted
                return
            if node.child_right is None:
                node.child_right = node_inserted
                return
        raise AssertionError

    def delete(self, value: ValueType) -> None:
        """Delete given value from tree.

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """

        def delete_node_and_return_replacement(root: _BinaryNode) -> _BinaryNode | None:
            assert root is not None
            if root.num_children == 0:
                # Leaf nodes can be simply removed
                return None
            if root.num_children == 1:
                # Replace node with its child
                return root.child_left or root.child_right

            # Find node with one leaf child
            leaf_parent = root
            while leaf_parent.child_left.num_children > 0 and leaf_parent.child_right.num_children > 0:
                leaf_parent = leaf_parent.child_left or leaf_parent.child_right

            # Replace root with leaf child
            if leaf_parent.child_left.num_children == 0:
                # Left child is leaf
                root.value = leaf_parent.child_left.value
                leaf_parent.child_left = None
            else:
                # Right child is leaf
                root.value = leaf_parent.child_right.value
                leaf_parent.child_right = None
            return root

        assert self._root is not None
        if self._root.value == value:
            self._root = delete_node_and_return_replacement(self._root)
            return
        for node in self._traversal_BFS():
            assert node.value != value
            if node.child_left.value == value:
                node.child_left = delete_node_and_return_replacement(node.child_left)
                return
            if node.child_right.value == value:
                node.child_right = delete_node_and_return_replacement(node.child_right)
                return
        raise AssertionError

    def traversal_BFS(self) -> Iterator[ValueType]:
        """Breadth-First Search tree traversal (level order).

        Applications
        ------------
        https://en.wikipedia.org/wiki/Breadth-first_search#Applications

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """
        for node in self._traversal_BFS():
            yield node.value

    def _traversal_BFS(self) -> Iterator[_BinaryNode]:
        if self._root is None:
            return
        queue_nodes = collections.deque([self._root])
        while queue_nodes:
            node = queue_nodes.popleft()
            if node.child_left is not None:
                queue_nodes.append(node.child_left)
            if node.child_right is not None:
                queue_nodes.append(node.child_right)
            yield node

    def traversal_DFS(self, variant: Literal["preorder", "inorder", "postorder"]) -> Iterator[ValueType]:
        """Depth-First Search tree traversal.

        It comes in the following flavors:
        - preorder:  root -> left  -> right
        - inorder:   left -> root  -> right
        - postorder: left -> right -> root

        Applications
        ------------
        https://en.wikipedia.org/wiki/Depth-first_search#Applications

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """
        match variant:
            case "preorder":
                method = self._preorder
            case "inorder":
                method = self._inorder
            case "postorder":
                method = self._postorder
            case _:
                raise AssertionError(f'Invalid DFS variant "{variant}"')
        for node in method(self._root):
            yield node.value

    @classmethod
    def _preorder(cls, root: _BinaryNode | None) -> Iterator[_BinaryNode]:
        """root-left-right traversal policy.

        Topologically sorted, since parents are processed before their children.

        Applications:
        - copying/cloning trees
        - prefix notation
        """
        if root is None:
            return
        yield root
        yield from cls._preorder(root.child_left)
        yield from cls._preorder(root.child_right)

    @classmethod
    def _inorder(cls, root: _BinaryNode | None) -> Iterator[_BinaryNode]:
        """left-root-right traversal policy.

        Applications:
        - sorting binary search tree
        - expression evaluation
        """

        if root is None:
            return
        yield from cls._inorder(root.child_left)
        yield root
        yield from cls._inorder(root.child_right)

    @classmethod
    def _postorder(cls, root: _BinaryNode | None) -> Iterator[_BinaryNode]:
        """left-right-root traversal policy.

        Applications:
        - deleting nodes
        - postfix notation
        - expression evaluation
        """

        if root is None:
            return
        yield from cls._postorder(root.child_left)
        yield from cls._postorder(root.child_right)
        yield root


@dataclasses.dataclass()
class _BinaryNode:
    value: ValueType
    child_left: Self | None = None
    child_right: Self | None = None

    @property
    def balance(self) -> int:
        height_left = self.child_left.height if self.child_left is not None else 0
        height_right = self.child_right.height if self.child_right is not None else 0
        return height_left - height_right

    @property
    def height(self) -> int:
        height_left = self.child_left.height if self.child_left is not None else 0
        height_right = self.child_right.height if self.child_right is not None else 0
        return 1 + max(height_left, height_right)

    @property
    def num_children(self) -> int:
        return 2 - [self.child_left, self.child_right].count(None)
