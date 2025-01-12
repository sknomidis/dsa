from __future__ import annotations

import pytest
from pytest import approx

from dsa.algorithms import shortest_path
from dsa.data_structures import graphs


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_two_vertices(algorithm: shortest_path.shortest_path_algorithm, directed: bool) -> None:
    # vertex_1 -- 2.0 -- vertex_2
    graph = graphs.Graph(representation="adjacency_matrix", directed=directed)
    graph.add_value("vertex_1")
    graph.add_value("vertex_2")
    graph.add_connection("vertex_1", "vertex_2", weight=2.0)

    path_and_distance = algorithm(graph, vertex_source="vertex_1", vertex_target="vertex_2")
    assert path_and_distance.path == ["vertex_1", "vertex_2"]
    assert path_and_distance.distance == approx(2.0)

    if directed:
        with pytest.raises(AssertionError, match="Source and target are not connected"):
            path_and_distance = algorithm(graph, vertex_source="vertex_2", vertex_target="vertex_1")
    else:
        path_and_distance = algorithm(graph, vertex_source="vertex_2", vertex_target="vertex_1")
        assert path_and_distance.path == ["vertex_2", "vertex_1"]
        assert path_and_distance.distance == approx(2.0)


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_4_vertices(algorithm: shortest_path.shortest_path_algorithm, directed: bool) -> None:
    #              vertex_2
    #            /          \
    #          1.0          1.5
    #          /              \
    # vertex_1                  vertex_4
    #          \              /
    #          2.0          1.0
    #            \          /
    #              vertex_3
    graph = graphs.Graph(representation="adjacency_matrix", directed=directed)
    graph.add_value("vertex_1")
    graph.add_value("vertex_2")
    graph.add_value("vertex_3")
    graph.add_value("vertex_4")
    graph.add_connection("vertex_1", "vertex_2", weight=1.0)
    graph.add_connection("vertex_2", "vertex_4", weight=1.5)
    graph.add_connection("vertex_1", "vertex_3", weight=2.0)
    graph.add_connection("vertex_3", "vertex_4", weight=1.0)

    path_and_distance = algorithm(graph, vertex_source="vertex_1", vertex_target="vertex_4")
    assert path_and_distance.path == ["vertex_1", "vertex_2", "vertex_4"]
    assert path_and_distance.distance == approx(2.5)

    if not directed:
        path_and_distance = algorithm(graph, vertex_source="vertex_3", vertex_target="vertex_2")
        assert path_and_distance.path == ["vertex_3", "vertex_4", "vertex_2"]
        assert path_and_distance.distance == approx(2.5)


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_7_vertices(algorithm: shortest_path.shortest_path_algorithm, directed: bool) -> None:
    #              vertex_3                  vertex_5
    #            /          \              /         \
    #          6.0          8.0          10.0        2.0
    #          /              \          /              \
    # vertex_1                  vertex_4                  vertex_7
    #          \              /          \              /
    #          2.0          5.0          15.0         6.0
    #            \          /              \          /
    #              vertex_2                  vertex_6
    graph = graphs.Graph(representation="adjacency_matrix", directed=directed)
    graph.add_value("vertex_1")
    graph.add_value("vertex_2")
    graph.add_value("vertex_3")
    graph.add_value("vertex_4")
    graph.add_value("vertex_5")
    graph.add_value("vertex_6")
    graph.add_value("vertex_7")
    graph.add_connection("vertex_1", "vertex_2", weight=2.0)
    graph.add_connection("vertex_1", "vertex_3", weight=6.0)
    graph.add_connection("vertex_2", "vertex_4", weight=5.0)
    graph.add_connection("vertex_3", "vertex_4", weight=8.0)
    graph.add_connection("vertex_4", "vertex_5", weight=10.0)
    graph.add_connection("vertex_4", "vertex_6", weight=15.0)
    graph.add_connection("vertex_5", "vertex_7", weight=2.0)
    graph.add_connection("vertex_6", "vertex_7", weight=6.0)

    path_and_distance = algorithm(graph, vertex_source="vertex_1", vertex_target="vertex_7")
    assert path_and_distance.path == ["vertex_1", "vertex_2", "vertex_4", "vertex_5", "vertex_7"]
    assert path_and_distance.distance == approx(19.0)

    if directed:
        path_and_distance = algorithm(graph, vertex_source="vertex_3", vertex_target="vertex_7")
        assert path_and_distance.path == ["vertex_3", "vertex_4", "vertex_5", "vertex_7"]
        assert path_and_distance.distance == approx(20.0)
    else:
        path_and_distance = algorithm(graph, vertex_source="vertex_7", vertex_target="vertex_2")
        assert path_and_distance.path == ["vertex_7", "vertex_5", "vertex_4", "vertex_2"]
        assert path_and_distance.distance == approx(17.0)
