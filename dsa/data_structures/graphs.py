from __future__ import annotations

import abc
import collections
import math
import numbers
from typing import Callable, Iterator, Literal, Sequence, TypeAlias, TypeVar

GraphElementType = TypeVar("GraphElementType")
RepresentationType: TypeAlias = Literal["adjacency_list", "adjacency_matrix", "pointers_and_objects"]


class Graph:
    def __init__(self, *, representation: RepresentationType, directed: bool) -> None:
        self._values: list[GraphElementType] = []
        self._directed = directed
        self._representation = self._get_representation(representation)

    @property
    def values(self) -> list[GraphElementType]:
        return self._values

    def __len__(self) -> int:
        return len(self._values)

    def traverse_BFS(self, value: GraphElementType) -> Iterator[GraphElementType]:
        """Breadth-First Search graph traversal.

        Traverse vertices level by level, starting with neighbors of a vertex,
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
            index = queue.popleft()
            yield self._values[index]
            for index_neighbor, _ in self._representation.iterate_neighbors(index):
                if not has_been_traversed[index_neighbor]:
                    has_been_traversed[index_neighbor] = True
                    queue.append(index_neighbor)

    def traverse_DFS(self, value: GraphElementType) -> Iterator[GraphElementType]:
        """Depth-First Search graph traversal.

        Traverse vertices one by one, starting with first neighbor of a vertex,
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

    def iterate_neighbors(self, value: GraphElementType) -> Iterator[tuple[GraphElementType, float]]:
        """Iterate through adjacent nodes."""
        index = self._get_index(value)
        for index_neighbor, weight in self._representation.iterate_neighbors(index):
            yield self._values[index_neighbor], weight

    def has_value(self, value: GraphElementType) -> bool:
        """Check if value exists in graph or not."""
        return value in self._values

    def add_value(self, value: GraphElementType) -> None:
        """Add value to graph."""
        assert not self.has_value(value)
        self._values.append(value)
        self._representation.add_vertex()

    def remove_value(self, value: GraphElementType) -> None:
        """Remove value from graph."""
        index = self._get_index(value)
        del self._values[index]
        self._representation.remove_vertex(index)

    def has_connection(self, value_1: GraphElementType, value_2: GraphElementType) -> bool:
        """Check if values are directly connected."""
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        if self._directed:
            return self._representation.has_edge(index_1, index_2)
        return self._representation.has_edge(index_1, index_2) or self._representation.has_edge(index_2, index_1)

    def get_connection_weight(self, value_1: GraphElementType, value_2: GraphElementType) -> float:
        """Get weight of connection between values."""
        assert self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        return self._representation.get_edge(index_1, index_2)

    def add_connection(self, value_1: GraphElementType, value_2: GraphElementType, *, weight: float = 1.0) -> None:
        """Add direct connection between values."""
        assert weight > 0.0
        assert not self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        self._representation.add_edge(index_1, index_2, weight=weight)
        if not self._directed:
            self._representation.add_edge(index_2, index_1, weight=weight)

    def remove_connection(self, value_1: GraphElementType, value_2: GraphElementType) -> None:
        """Remove direct connection between values."""
        assert self.has_connection(value_1, value_2)
        index_1 = self._get_index(value_1)
        index_2 = self._get_index(value_2)
        self._representation.remove_edge(index_1, index_2)
        if not self._directed:
            self._representation.remove_edge(index_2, index_1)

    def _get_index(self, value: GraphElementType) -> int:
        assert self.has_value(value)
        for index, value_stored in enumerate(self._values):
            if value_stored == value:
                return index

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


_VertexAndWeight = collections.namedtuple("_VertexAndWeight", ["vertex", "weight"])


class _GraphRepresentation(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_vertices(self) -> int: ...

    @abc.abstractmethod
    def iterate_neighbors(self, vertex: int) -> Iterator[_VertexAndWeight]: ...

    @abc.abstractmethod
    def add_vertex(self) -> None: ...

    @abc.abstractmethod
    def remove_vertex(self, vertex: int) -> None: ...

    @abc.abstractmethod
    def has_edge(self, vertex_1: int, vertex_2: int) -> bool: ...

    @abc.abstractmethod
    def get_edge(self, vertex_1: int, vertex_2: int) -> float: ...

    @abc.abstractmethod
    def add_edge(self, vertex_1: int, vertex_2: int, *, weight: float) -> None: ...

    @abc.abstractmethod
    def remove_edge(self, vertex_1: int, vertex_2: int) -> None: ...


class _AdjacencyList(_GraphRepresentation):
    """Adjacency List graph representation.

    Uses a list of lists to keep track of edges. Each row contains the neighbors
    indices for that vertex.

    Complexity
    ----------
    Space: O(V + E)
    """

    def __init__(self) -> None:
        self._adjacency_list: list[list[_VertexAndWeight]] = []

    @property
    def n_vertices(self) -> int:
        return len(self._adjacency_list)

    def iterate_neighbors(self, vertex: int) -> Iterator[_VertexAndWeight]:
        # Time complexity: O(E)
        assert 0 <= vertex < self.n_vertices
        yield from self._adjacency_list[vertex]

    def add_vertex(self) -> None:
        # Time complexity: O(1)
        self._adjacency_list.append([])

    def remove_vertex(self, vertex: int) -> None:
        # Time complexity: O(1)
        assert vertex < self.n_vertices
        del self._adjacency_list[vertex]
        for neighbors in self._adjacency_list:
            for vertex_and_weight in neighbors:
                if vertex_and_weight.vertex == vertex:
                    neighbors.remove(vertex_and_weight)
                    break

    def has_edge(self, vertex_1: int, vertex_2: int) -> bool:
        # Time complexity: O(E)
        assert vertex_1 != vertex_2
        assert 0 <= vertex_1 < self.n_vertices
        assert 0 <= vertex_2 < self.n_vertices
        for vertex_and_weight in self._adjacency_list[vertex_1]:
            if vertex_and_weight.vertex == vertex_2:
                return True
        return False

    def get_edge(self, vertex_1: int, vertex_2: int) -> float:
        assert vertex_1 != vertex_2
        assert 0 <= vertex_1 < self.n_vertices
        assert 0 <= vertex_2 < self.n_vertices
        for vertex_and_weight in self._adjacency_list[vertex_1]:
            if vertex_and_weight.vertex == vertex_2:
                return vertex_and_weight.weight
        return math.inf

    def add_edge(self, vertex_1: int, vertex_2: int, *, weight: float) -> None:
        # Time complexity: O(E)
        assert vertex_1 != vertex_2
        assert 0 <= vertex_1 < self.n_vertices
        assert 0 <= vertex_2 < self.n_vertices
        vertex_and_weight = _VertexAndWeight(vertex_2, weight)
        _add_element_sorted(self._adjacency_list[vertex_1], vertex_and_weight, key=lambda t: t.vertex)

    def remove_edge(self, vertex_1: int, vertex_2: int) -> None:
        # Time complexity: O(E)
        assert vertex_1 != vertex_2
        assert 0 <= vertex_1 < self.n_vertices
        assert 0 <= vertex_2 < self.n_vertices
        for vertex_and_weight in self._adjacency_list[vertex_1]:
            if vertex_and_weight.vertex == vertex_2:
                self._adjacency_list[vertex_1].remove(vertex_and_weight)


class _AdjacencyMatrix(_GraphRepresentation):
    """Adjacency Matrix graph representation.

    Uses a VxV matrix to keep track of edges. The ij-th element is `True` if
    vertex i is connected to vertex j, and `False` otherwise.

    Complexity
    ----------
    Space: O(V**2)
    """

    def __init__(self) -> int:
        self._adjacency_matrix: list[list[float]] = []

    @property
    def n_vertices(self) -> int:
        return len(self._adjacency_matrix)

    def iterate_neighbors(self, vertex: int) -> Iterator[_VertexAndWeight]:
        # Time complexity: O(V)
        for neighbor, weight in enumerate(self._adjacency_matrix[vertex]):
            if weight > 0.0:
                yield _VertexAndWeight(neighbor, weight)

    def add_vertex(self) -> None:
        # Time complexity: O(V**2)
        self._adjacency_matrix.append(self.n_vertices * [0.0])
        for row in self._adjacency_matrix:
            row.append(0.0)

    def remove_vertex(self, vertex: int) -> None:
        # Time complexity: O(V**2)
        del self._adjacency_matrix[vertex]
        for row in self._adjacency_matrix:
            del row[vertex]

    def has_edge(self, vertex_1: int, vertex_2: int) -> bool:
        # Time complexity: O(1)
        return self._adjacency_matrix[vertex_1][vertex_2] > 0.0

    def get_edge(self, vertex_1: int, vertex_2: int) -> float:
        return self._adjacency_matrix[vertex_1][vertex_2]

    def add_edge(self, vertex_1: int, vertex_2: int, *, weight: float) -> None:
        # Time complexity: O(1)
        assert weight > 0.0
        self._adjacency_matrix[vertex_1][vertex_2] = weight

    def remove_edge(self, vertex_1: int, vertex_2: int) -> None:
        # Time complexity: O(1)
        self._adjacency_matrix[vertex_1][vertex_2] = 0.0


class _PointersAndObjects(_GraphRepresentation):
    """Pointers and Objects graph representation.

    Uses a pointer list of length V to keep track of vertices. Each `Vertex`
    is an object that holds a list of its neighbors.

    Complexity
    ----------
    Space: O(V + E)
    """

    def __init__(self) -> int:
        self._vertices: list[_Vertex] = []

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    def iterate_neighbors(self, vertex: int) -> Iterator[_VertexAndWeight]:
        # Time complexity: O(E)
        for vertex_and_neighbor in self._vertices[vertex]:
            yield _VertexAndWeight(vertex_and_neighbor.vertex.index, vertex_and_neighbor.weight)

    def add_vertex(self) -> None:
        # Time complexity: O(1)
        self._vertices.append(_Vertex(index=self.n_vertices))

    def remove_vertex(self, vertex: int) -> None:
        # Time complexity: O(V)
        self._vertices.pop(vertex)
        for index, vertix in enumerate(self._vertices):
            vertix.index = index

    def has_edge(self, vertex_1: int, vertex_2: int) -> bool:
        # Time complexity: O(E)
        return self._vertices[vertex_1].has_neighbor(self._vertices[vertex_2])

    def get_edge(self, vertex_1: int, vertex_2: int) -> float:
        return self._vertices[vertex_1].get_neighbor_weight(vertex_2)

    def add_edge(self, vertex_1: int, vertex_2: int, *, weight: float) -> None:
        # Time complexity: O(E)
        self._vertices[vertex_1].add_neighbor(self._vertices[vertex_2], weight=weight)

    def remove_edge(self, vertex_1: int, vertex_2: int) -> None:
        # Time complexity: O(E)
        self._vertices[vertex_1].remove_neighbor(self._vertices[vertex_2])


class _Vertex:
    def __init__(self, index: int) -> None:
        self.index = index
        self._neighbors: list[_VertexAndWeight] = []

    def __iter__(self) -> Iterator[_VertexAndWeight]:
        # Time complexity: O(E)
        yield from self._neighbors

    def has_neighbor(self, vertex: _Vertex) -> bool:
        # Time complexity: O(E)
        assert vertex is not self
        return any(vertex is neighbor.vertex for neighbor in self._neighbors)

    def get_neighbor_weight(self, vertex: _Vertex) -> float:
        assert self.has_neighbor(vertex)
        for vertex_and_weight in self:
            if vertex_and_weight.vertex is vertex:
                return vertex_and_weight.weight
        raise AssertionError

    def add_neighbor(self, vertex: _Vertex, *, weight: float) -> None:
        # Time complexity: O(E)
        assert not self.has_neighbor(vertex)
        _add_element_sorted(self._neighbors, _VertexAndWeight(vertex, weight), key=lambda t: getattr(t.vertex, "index"))

    def remove_neighbor(self, vertex: _Vertex) -> None:
        # Time complexity: O(E)
        for vertex_and_weight in self:
            self._neighbors.remove(vertex_and_weight)
            return
        raise AssertionError


_T = TypeVar("_T")


def _add_element_sorted(sequence: Sequence[_T], element: _T, *, key: Callable[[_T], numbers.Number]) -> None:
    index_to_insert = 0
    for index, neighbor in enumerate(sequence):
        if key(neighbor) == key(element):
            return
        if key(neighbor) < key(element):
            index_to_insert = index + 1
    sequence.insert(index_to_insert, element)
