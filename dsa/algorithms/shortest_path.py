from __future__ import annotations

import collections
import heapq
import math
from typing import Callable, NamedTuple, Protocol

from dsa.data_structures import graphs


class PathAndDistance(NamedTuple):
    path: list[graphs.ValueType]
    distance: graphs.WeightType


class ShortestPathAlgorithm(Protocol):
    """Find shortest path and distance between two nodes in a weighted graph."""

    def __call__(self, graph: graphs.Graph, source: graphs.ValueType, target: graphs.ValueType) -> PathAndDistance: ...


def no_guess_min_distance_from_target(
    graph: graphs.Graph, source: graphs.ValueType, target: graphs.ValueType
) -> graphs.WeightType:
    # When no educated guess can be made for the distance between `source` and
    # `target`, the A* search algorithm reduces to Dijkstra's algorithm.
    return 0.0


def a_star_search_algorithm(
    graph: graphs.Graph,
    source: graphs.ValueType,
    target: graphs.ValueType,
    guess_min_distance_from_target: Callable[
        [graphs.Graph, graphs.ValueType, graphs.ValueType], graphs.WeightType
    ] = no_guess_min_distance_from_target,
) -> PathAndDistance:
    """A* search algorithm for finding shortest path between two nodes.

    Logic
    -----
    The algorithm consists of the following steps:
    1. Set current node to source node
    2. Go through neighboring nodes not yet visited, and:
      a. For each neighbor compute total distance from source to neighbor,
         with current node as the second to last node of the path
      b. Add the estimated distance between the neighbor and the target
      c. If total distance is smaller than previously stored one, this becomes
         the new shortest path from source to neighbor
    3. Find node with shortest distance out of all unvisited nodes, and
       move to that one
    4. Repeat 2-3 until target is encountered. Then compute and return shortest
       path and distance.

    `guess_min_distance_from_target` is a problem-specific heuristic function
    for estimating the distance of a given node from the target node. If it
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
    unvisited_nodes = [(0.0, source)]
    # Used to reconstruct the final path
    node_to_min_previous_node: dict[graphs.ValueType, graphs.ValueType] = {}
    # g-score: computed min distance of a node from the source
    node_to_min_distance_from_source = collections.defaultdict(lambda: math.inf)
    node_to_min_distance_from_source[source] = 0.0
    # f-score: estimated min distance from source to target through node
    node_to_min_distance_from_source_to_target = collections.defaultdict(lambda: math.inf)
    node_to_min_distance_from_source_to_target[source] = guess_min_distance_from_target(graph, source, target)

    while unvisited_nodes:
        # Step 2: Move to unvisited node with smallest estimated distance
        # between source and target
        _, current = heapq.heappop(unvisited_nodes)

        # Stop if target has been visited
        if current == target:
            break

        # Step 3: Update distance for neighboring unvisited nodes, if smaller
        distance_source_to_current = node_to_min_distance_from_source[current]
        for neighbor, distance_current_to_neighbor in graph.iterate_neighbors(current):
            distance_source_to_neighbor = distance_source_to_current + distance_current_to_neighbor
            if distance_source_to_neighbor < node_to_min_distance_from_source[neighbor]:
                node_to_min_previous_node[neighbor] = current
                node_to_min_distance_from_source[neighbor] = distance_source_to_neighbor
                node_to_min_distance_from_source_to_target[neighbor] = (
                    distance_source_to_neighbor + guess_min_distance_from_target(graph, neighbor, target)
                )
                heapq.heappush(
                    unvisited_nodes,
                    (node_to_min_distance_from_source_to_target[neighbor], neighbor),
                )

    assert current == target, "Source and target are not connected"

    # Step 4: Recover shortest path and distance
    shortest_path = [target]
    current = target
    while current in node_to_min_previous_node:
        current = node_to_min_previous_node[current]
        shortest_path.insert(0, current)
    shortest_distance = node_to_min_distance_from_source[target]
    return PathAndDistance(shortest_path, shortest_distance)
