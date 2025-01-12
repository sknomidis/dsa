from __future__ import annotations

import pytest
from pytest import approx

from dsa.algorithms import traveling_salesman
from dsa.data_structures import graphs


@pytest.mark.parametrize(
    "algorithm",
    [traveling_salesman.traveling_salesman_brute_force, traveling_salesman.traveling_salesman_recursion],
    ids=lambda f: f.__name__,
)
def test_traveling_salesman(algorithm: traveling_salesman.TravelingSalesmanAlgorithm) -> None:
    graph = graphs.Graph(directed=False)
    graph.add_value("city_1")
    graph.add_value("city_2")
    graph.add_value("city_3")
    graph.add_value("city_4")
    graph.add_connection("city_1", "city_2", weight=10.0)
    graph.add_connection("city_1", "city_3", weight=15.0)
    graph.add_connection("city_1", "city_4", weight=20.0)
    graph.add_connection("city_2", "city_3", weight=35.0)
    graph.add_connection("city_2", "city_4", weight=25.0)
    graph.add_connection("city_3", "city_4", weight=30.0)

    path, distance = algorithm(graph)

    assert path == ["city_1", "city_2", "city_4", "city_3", "city_1"]
    assert distance == approx(80.0)
