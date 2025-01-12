from __future__ import annotations

import itertools
import math
from typing import Protocol

from dsa.data_structures import graphs


class TravelingSalesmanAlgorithm(Protocol):
    """Solution to the Traveling Salesman Problem (TSP).

    Given a list of cities and the distances between each pair of cities, what
    is the shortest possible route that visits each city exactly once and
    returns to the origin city?

    The input to the problem is a complete graph, meaning there is an edge
    between any pair of vertices.
    """

    def __call__(self, graph: graphs.Graph) -> tuple[list[graphs.GraphElementType], float]: ...


def traveling_salesman_brute_force(graph: graphs.Graph) -> tuple[list[graphs.GraphElementType], float]:
    """Brute force approach to the TSP.

    Consider all possible paths, and compute the corresponding distance. Select
    and return path with smallest total distance.

    Complexity
    ----------
    Time: O(N!)
    Space: O(N)
    """

    distance_min = math.inf
    path_min: list[graphs.GraphElementType] = []

    for cities in itertools.permutations(graph.values):
        # We need to return to starting city
        cities = list(cities) + [cities[0]]

        # Compute total distance
        distance = 0.0
        for city_current, city_next in itertools.pairwise(cities):
            distance += graph.get_connection_weight(city_current, city_next)

        # Store if smallest so far
        if distance < distance_min:
            distance_min = distance
            path_min = list(cities)

    return path_min, distance_min


def traveling_salesman_recursion(graph: graphs.Graph) -> tuple[list[graphs.GraphElementType], float]:
    """Recursion approach to the TSP.

    Recursively go through all possible city permutations, by considering all
    possible next stops for a given path, and choosing the one with the smallest
    extra distance.

    Complexity
    ----------
    Time: O(N!)
    Space: O(N)
    """
    cities_all = graph.values
    city_start = cities_all[0]

    def compute_min_path_and_distance(
        city_current: graphs.GraphElementType, path_covered: list[graphs.GraphElementType], distance_covered: float
    ) -> tuple[list[graphs.GraphElementType], float]:
        if len(path_covered) == len(cities_all):
            # Return to starting city
            path_covered.append(city_start)
            distance_covered += graph.get_connection_weight(city_current, city_start)
            return path_covered, distance_covered

        distance_min = math.inf
        path_min: list[graphs.GraphElementType] = []
        for city_next in cities_all:
            # Go through all unvisited cities
            if city_next in path_covered:
                continue

            # Compute minimum remaining path and distance through that stop
            path, distance = compute_min_path_and_distance(
                city_current=city_next,
                path_covered=path_covered + [city_next],
                distance_covered=distance_covered + graph.get_connection_weight(city_current, city_next),
            )

            # Store if smallest so far
            if distance < distance_min:
                distance_min = distance
                path_min = path

        return path_min, distance_min

    return compute_min_path_and_distance(city_start, path_covered=[city_start], distance_covered=0.0)
