from __future__ import annotations

import abc
import collections
import math
from typing import Iterator, Literal, NamedTuple, Self, TypeAlias, TypeVar

RepresentationType: TypeAlias = Literal["adjacency_list", "adjacency_matrix", "pointers_and_objects"]

ValueType = TypeVar("ValueType")
WeightType: TypeAlias = float


class Graph:
    """A collection of nodes connected by edges."""

    def __init__(self, *, representation: RepresentationType = "adjacency_matrix", directed: bool = True) -> None:
        self._representation = self._get_representation(representation)
        self._directed = directed
        self._values: list[ValueType] = []

    @property
    def values(self) -> list[ValueType]:
        return self._values

    def __len__(self) -> int:
        return len(self._values)

    def traverse_BFS(self, value: ValueType) -> Iterator[ValueType]:
        """Breadth-First Search graph traversal.

        Traverse nodes level by level, starting with neighbors of a node,
        then their neighbors, etc.

        Extra care should be taken to avoid cycles.

        Complexity
        ----------
        Time: O(V + E)
        Space: O(V)
        """
        index = self._get_index(value)
        has_been_traversed = len(self) * [False]
        has_been_traversed[index] = True

        queue = collections.deque([index])
        while queue:
            index = queue.pop()
            yield self._values[index]
            for index_neighbor, _ in self._representation.iterate_neighbors(index):
                if not has_been_traversed[index_neighbor]:
                    has_been_traversed[index_neighbor] = True
                    queue.insert(0, index_neighbor)

    def traverse_DFS(self, value: ValueType) -> Iterator[ValueType]:
        """Depth-First Search graph traversal.

        Traverse nodes one by one, starting with first neighbor of a node,
        then its first neighbor, etc.

        Extra care should be taken to avoid cycles.

        Complexity
        ----------
        Time: O(V + E)
        Space: O(V + E)
        """
        has_been_traversed = len(self) * [False]

        def traverse(index: int) -> Iterator[int]:
            if has_been_traversed[index]:
                return
            yield index
            has_been_traversed[index] = True
            for index_neighbor, _ in self._representation.iterate_neighbors(index):
                yield from traverse(index_neighbor)

        for index in traverse(self._get_index(value)):
            yield self._values[index]

    def iterate_neighbors(self, value: ValueType) -> Iterator[tuple[ValueType, WeightType]]:
        """Iterate through adjacent nodes and correpsonding edge weights.

        Complexity
        ----------
        Time: O(V + E)
        Space: O(1)
        """
        index = self._get_index(value)
        for index_neighbor, weight in self._representation.iterate_neighbors(index):
            yield self._values[index_neighbor], weight

    def has_value(self, value: ValueType) -> bool:
        """Check if value exists in graph or not."""
        return value in self._values

    def add_value(self, value: ValueType) -> None:
        """Add value to graph."""
        assert not self.has_value(value)
        self._values.append(value)
        self._representation.add_node()

    def delete_value(self, value: ValueType) -> None:
        """Delete value from graph."""
        index = self._get_index(value)
        del self._values[index]
        self._representation.delete_node(index)

    def has_connection(self, value_1: ValueType, value_2: ValueType) -> bool:
        """Check if values are directly connected."""
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        return self._representation.has_edge(index_1, index_2)

    def get_connection_weight(self, value_1: ValueType, value_2: ValueType) -> WeightType:
        """Get weight of connection between values."""
        assert self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        return self._representation.get_edge(index_1, index_2)

    def add_connection(self, value_1: ValueType, value_2: ValueType, *, weight: WeightType = 1.0) -> None:
        """Add direct connection between values."""
        assert weight > 0.0
        assert not self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        self._representation.add_edge(index_1, index_2, weight=weight)
        if not self._directed:
            self._representation.add_edge(index_2, index_1, weight=weight)

    def delete_connection(self, value_1: ValueType, value_2: ValueType) -> None:
        """Delete direct connection between values."""
        assert self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        self._representation.delete_edge(index_1, index_2)
        if not self._directed:
            self._representation.delete_edge(index_2, index_1)

    def _get_index(self, value: ValueType) -> int:
        assert self.has_value(value)
        return self._values.index(value)

    @staticmethod
    def _get_representation(representation: RepresentationType) -> _GraphRepresentation:
        match representation:
            case "adjacency_list":
                return _AdjacencyList()
            case "adjacency_matrix":
                return _AdjacencyMatrix()
            case "pointers_and_objects":
                return _PointersAndObjects()
            case _:
                raise AssertionError


class _NodeAndWeight(NamedTuple):
    index: int
    weight: WeightType


class _GraphRepresentation(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> int: ...

    @property
    @abc.abstractmethod
    def num_nodes(self) -> int: ...

    @abc.abstractmethod
    def iterate_neighbors(self, index: int) -> Iterator[_NodeAndWeight]: ...

    @abc.abstractmethod
    def add_node(self) -> None: ...

    @abc.abstractmethod
    def delete_node(self, index: int) -> None: ...

    @abc.abstractmethod
    def has_edge(self, index_1: int, index_2: int) -> bool: ...

    @abc.abstractmethod
    def get_edge(self, index_1: int, index_2: int) -> WeightType: ...

    @abc.abstractmethod
    def add_edge(self, index_1: int, index_2: int, *, weight: WeightType) -> None: ...

    @abc.abstractmethod
    def delete_edge(self, index_1: int, index_2: int) -> None: ...


class _AdjacencyList(_GraphRepresentation):
    """Adjacency List graph representation.

    Uses a list of lists to keep track of edges. Each row contains the neighbors
    indices for that node.

    Ideal for sparse graphs.

    Complexity
    ----------
    Space: O(V + E)
    """

    def __init__(self) -> None:
        self._adjacency_list: list[list[_NodeAndWeight]] = []

    @property
    def num_nodes(self) -> int:
        return len(self._adjacency_list)

    def iterate_neighbors(self, index: int) -> Iterator[_NodeAndWeight]:
        # O(E)
        yield from self._adjacency_list[index]

    def add_node(self) -> None:
        # O(1)
        self._adjacency_list.append([])

    def delete_node(self, index: int) -> None:
        # O(N + E)
        for index_neighbor in range(len(self._adjacency_list)):
            if index_neighbor != index:
                self.delete_edge(index_neighbor, index)
        del self._adjacency_list[index]

    def has_edge(self, index_1: int, index_2: int) -> bool:
        # O(E)
        return any(index_and_weight.index == index_2 for index_and_weight in self._adjacency_list[index_1])

    def get_edge(self, index_1: int, index_2: int) -> WeightType:
        # O(E)
        for index_and_weight in self._adjacency_list[index_1]:
            if index_and_weight.index == index_2:
                return index_and_weight.weight
        return math.inf

    def add_edge(self, index_1: int, index_2: int, *, weight: WeightType) -> None:
        # O(E)
        self._adjacency_list[index_1].append(_NodeAndWeight(index_2, weight))
        self._adjacency_list[index_1].sort(key=lambda t: t.index)

    def delete_edge(self, index_1: int, index_2: int) -> None:
        # O(E)
        for index_and_weight in self._adjacency_list[index_1]:
            if index_and_weight.index == index_2:
                self._adjacency_list[index_1].remove(index_and_weight)


class _AdjacencyMatrix(_GraphRepresentation):
    """Adjacency Matrix graph representation.

    Uses a VxV matrix to keep track of edges. The ij-th element is `True` if
    node i is connected to node j, and `False` otherwise.

    Ideal for dense graphs.

    Complexity
    ----------
    Space: O(V**2)
    """

    def __init__(self) -> int:
        self._adjacency_matrix: list[list[WeightType]] = []

    @property
    def num_nodes(self) -> int:
        return len(self._adjacency_matrix)

    def iterate_neighbors(self, index: int) -> Iterator[_NodeAndWeight]:
        # O(V)
        for index_neighbor, weight in enumerate(self._adjacency_matrix[index]):
            if weight > 0.0:
                yield _NodeAndWeight(index_neighbor, weight)

    def add_node(self) -> None:
        # O(V**2)
        self._adjacency_matrix.append(self.num_nodes * [0.0])
        for row in self._adjacency_matrix:
            row.append(0.0)

    def delete_node(self, index: int) -> None:
        # O(V**2)
        del self._adjacency_matrix[index]
        for row in self._adjacency_matrix:
            del row[index]

    def has_edge(self, index_1: int, index_2: int) -> bool:
        # O(1)
        return self._adjacency_matrix[index_1][index_2] > 0.0

    def get_edge(self, index_1: int, index_2: int) -> WeightType:
        return self._adjacency_matrix[index_1][index_2]

    def add_edge(self, index_1: int, index_2: int, *, weight: WeightType) -> None:
        # O(1)
        assert weight > 0.0
        self._adjacency_matrix[index_1][index_2] = weight

    def delete_edge(self, index_1: int, index_2: int) -> None:
        # O(1)
        self._adjacency_matrix[index_1][index_2] = 0.0


class _PointersAndObjects(_GraphRepresentation):
    """Pointers and Objects graph representation.

    Uses a pointer list of length V to keep track of nodes. Each `Node`
    is an object that holds a list of its neighbors.

    Complexity
    ----------
    Space: O(V + E)
    """

    def __init__(self) -> int:
        self._nodes: list[_Node] = []

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def iterate_neighbors(self, index: int) -> Iterator[_NodeAndWeight]:
        # O(E)
        for neighbor, weight in self._nodes[index]:
            yield _NodeAndWeight(neighbor.index, weight)

    def add_node(self) -> None:
        # O(1)
        self._nodes.append(_Node(index=self.num_nodes))

    def delete_node(self, index: int) -> None:
        # O(V)
        del self._nodes[index]
        for index, node in enumerate(self._nodes):
            node.index = index

    def has_edge(self, index_1: int, index_2: int) -> bool:
        # O(E)
        return self._nodes[index_1].has_neighbor(self._nodes[index_2])

    def get_edge(self, index_1: int, index_2: int) -> WeightType:
        # O(E)
        return self._nodes[index_1].get_neighbor_weight(index_2)

    def add_edge(self, index_1: int, index_2: int, *, weight: WeightType) -> None:
        # O(E)
        self._nodes[index_1].add_neighbor(self._nodes[index_2], weight=weight)

    def delete_edge(self, index_1: int, index_2: int) -> None:
        # O(E)
        self._nodes[index_1].delete_neighbor(self._nodes[index_2])


class _Node:
    def __init__(self, index: int) -> None:
        self.index = index
        self._neighbors: list[tuple[Self, WeightType]] = []

    def __iter__(self) -> Iterator[tuple[Self, WeightType]]:
        # O(E)
        yield from self._neighbors

    def has_neighbor(self, node: _Node) -> bool:
        # O(E)
        assert node is not self
        return any(node is neighbor for neighbor, _ in self._neighbors)

    def get_neighbor_weight(self, node: _Node) -> WeightType:
        # O(E)
        assert self.has_neighbor(node)
        for neighbor, weight in self:
            if neighbor is node:
                return weight
        raise AssertionError

    def add_neighbor(self, node: _Node, *, weight: WeightType) -> None:
        # O(E)
        assert not self.has_neighbor(node)
        self._neighbors.append((node, weight))
        self._neighbors.sort(key=lambda t: t[0].index)

    def delete_neighbor(self, node: _Node) -> None:
        # O(E)
        for neighbor, weight in self:
            if neighbor is node:
                self._neighbors.remove((neighbor, weight))
                return
        raise AssertionError
