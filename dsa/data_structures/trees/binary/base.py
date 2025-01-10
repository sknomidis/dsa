from __future__ import annotations

import collections
import dataclasses
from typing import Any, Iterator, Literal, Self


@dataclasses.dataclass()
class BinaryNode:
    value: Any
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
    def n_children(self) -> int:
        return 2 - [self.child_left, self.child_right].count(None)


class BinaryTree:
    def __init__(self, root: BinaryNode | None = None) -> None:
        self._root = root

    def is_balanced(self) -> bool:
        """Height difference between all left and right subtrees is at most 1."""

        def is_subtree_balanced(root: BinaryNode) -> bool:
            if root is None:
                return True
            height_left = root.child_left.height if root.child_left is not None else 0
            height_right = root.child_right.height if root.child_right is not None else 0
            skew = abs(height_left - height_right)
            return skew <= 1 and is_subtree_balanced(root.child_left) and is_subtree_balanced(root.child_right)

        return is_subtree_balanced(self._root)

    def is_complete(self) -> bool:
        """All intermediate levels are filled and all leaf nodes are on the left."""
        nonfull_node_has_been_found = False
        for node in self._traversal_BFS():
            n_children = node.n_children
            if nonfull_node_has_been_found:
                # All subsequent children must be leaf nodes
                if n_children > 0:
                    return False
                continue
            if n_children < 2:
                nonfull_node_has_been_found = True
                if n_children == 1 and node.child_right is not None:
                    return False
        return True

    def is_degenerate(self) -> bool:
        """Each node has 1 child."""
        if self._root is None or self._root.n_children != 1:
            return False
        for node in self._traversal_BFS():
            if node.n_children > 1:
                return False
        return True

    def is_full(self) -> bool:
        """Each node has either 0 or 2 children."""
        if self._root is None:
            return True
        for node in self._traversal_BFS():
            if node.n_children == 1:
                return False
        return True

    def is_perfect(self) -> bool:
        """Internal nodes have 2 children, and leaf nodes are on same level."""

        def is_subtree_perfect(root: BinaryNode, height: int) -> bool:
            if root is None:
                return True
            if root.n_children == 1:
                return False
            if root.n_children == 0:
                # Leaf nodes can only exist at last level
                return height == 1
            is_subtree_perfect_left = is_subtree_perfect(root.child_left, height - 1)
            is_subtree_perfect_right = is_subtree_perfect(root.child_right, height - 1)
            return is_subtree_perfect_left and is_subtree_perfect_right

        height_total = self._root.height if self._root is not None else 0
        return is_subtree_perfect(self._root, height_total)

    def insert(self, value: Any) -> None:
        """Breadth-First Search tree insertion (level order).

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """
        node_inserted = BinaryNode(value)
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

    def traversal_BFS(self) -> Iterator[Any]:
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

    def _traversal_BFS(self) -> Iterator[BinaryNode]:
        if self._root is None:
            return
        queue_children = collections.deque([self._root])
        while queue_children:
            node = queue_children.popleft()
            if node.child_left is not None:
                queue_children.append(node.child_left)
            if node.child_right is not None:
                queue_children.append(node.child_right)
            yield node

    def traversal_DFS(self, variant: Literal["preorder", "inorder", "postorder"]) -> Iterator[Any]:
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
        # Depth-first search traversal. Time and space complexity O(N)
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
    def _preorder(cls, root: BinaryNode | None) -> Iterator[BinaryNode]:
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
    def _inorder(cls, root: BinaryNode | None) -> Iterator[BinaryNode]:
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
    def _postorder(cls, root: BinaryNode | None) -> Iterator[BinaryNode]:
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
