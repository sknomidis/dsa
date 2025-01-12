from __future__ import annotations

import collections
import heapq
import math
from typing import Callable, NamedTuple, Protocol

from dsa.data_structures import graphs


class PathAndDistance(NamedTuple):
    path: list[graphs.GraphElementType]
    distance: float


class shortest_path_algorithm(Protocol):
    def __call__(
        self, graph: graphs.Graph, *, vertex_source: graphs.GraphElementType, vertex_target: graphs.GraphElementType
    ) -> PathAndDistance: ...


def no_guess_min_distance_from_target(
    graph: graphs.Graph, source: graphs.GraphElementType, target: graphs.GraphElementType
) -> float:
    # When no educated guess can be made for the distance between `source` and
    # `target`, the A* search algorithm reduces to Dijkstra's algorithm.
    return 0.0


def a_star_search_algorithm(
    graph: graphs.Graph,
    *,
    vertex_source: graphs.GraphElementType,
    vertex_target: graphs.GraphElementType,
    guess_min_distance_from_target: Callable[
        [graphs.Graph, graphs.GraphElementType, graphs.GraphElementType], float
    ] = no_guess_min_distance_from_target,
) -> PathAndDistance:
    """A* search algorithm for finding shortest path between two vertices.

    Logic
    -----
    The algorithm consists of the following steps:
    1. Set current vertex to source vertex
    2. Go through neighboring vertices not yet visited, and:
      a. For each neighbor compute total distance from source to neighbor,
         with current vertex as the second to last vertex of the path
      b. Add the estimated distance between the neighbor and the target
      c. If total distance is smaller than previously stored one, this becomes
         the new shortest path from source to neighbor
    3. Find vertex with shortest distance out of all unvisited vertices, and
       move to that one
    4. Repeat 2-3 until target is encountered. Then compute and return shortest
       path and distance.

    `guess_min_distance_from_target` is a problem-specific heuristic function
    for estimating the distance of a given vertex from the target vertex. If it
    is admissible, i.e., it never overestimates the actual distance, it is
    guaranteed to lead to the right solution.

    In the special case where it is a constant (no guessing), the A* algorithm
    reduces to Dijkstra's algorithm.

    Complexity
    ----------
    Time: O[(E + V) log V], thanks to min-priority queue
    Space: O(V)
    """
    # Step 1: Initialize
    # We use a min-priority queue for efficiency (original uses set)
    unvisited_vertices_min_priority_queue = [(0.0, vertex_source)]
    # Used to reconstruct the final path
    vertex_to_min_previous_vertex: dict[graphs.GraphElementType, graphs.GraphElementType] = {}
    # g-score: minimum computed distance of a vertex from the source
    vertex_to_min_distance_from_source = collections.defaultdict(lambda: math.inf)
    vertex_to_min_distance_from_source[vertex_source] = 0.0
    # f-score: minimum estimated distance of a vertex from the source
    vertex_to_min_distance_from_source_to_target = collections.defaultdict(lambda: math.inf)
    vertex_to_min_distance_from_source_to_target[vertex_source] = guess_min_distance_from_target(
        graph, vertex_source, vertex_target
    )

    while unvisited_vertices_min_priority_queue:
        # Step 2: Move to unvisited vertex with smallest estimated distance
        # between source and target
        _, vertex_current = heapq.heappop(unvisited_vertices_min_priority_queue)

        # Stop if target has been visited
        if vertex_current == vertex_target:
            break

        # Step 3: Update distance for neighboring unvisited vertices, if smaller
        distance_source_to_current = vertex_to_min_distance_from_source[vertex_current]
        for vertex_neighbor, distance_current_to_neighbor in graph.iterate_neighbors(vertex_current):
            distance_source_to_neighbor = distance_source_to_current + distance_current_to_neighbor
            if distance_source_to_neighbor < vertex_to_min_distance_from_source[vertex_neighbor]:
                vertex_to_min_previous_vertex[vertex_neighbor] = vertex_current
                vertex_to_min_distance_from_source[vertex_neighbor] = distance_source_to_neighbor
                vertex_to_min_distance_from_source_to_target[vertex_neighbor] = (
                    distance_source_to_neighbor + guess_min_distance_from_target(graph, vertex_neighbor, vertex_target)
                )
                heapq.heappush(
                    unvisited_vertices_min_priority_queue,
                    (vertex_to_min_distance_from_source_to_target[vertex_neighbor], vertex_neighbor),
                )

    assert vertex_current == vertex_target, "Source and target are not connected"

    # Step 4: Recover shortest path and distance
    shortest_path = [vertex_target]
    vertex_current = vertex_target
    while vertex_current in vertex_to_min_previous_vertex:
        vertex_current = vertex_to_min_previous_vertex[vertex_current]
        shortest_path.insert(0, vertex_current)
    shortest_distance = vertex_to_min_distance_from_source[vertex_target]
    return PathAndDistance(shortest_path, shortest_distance)
