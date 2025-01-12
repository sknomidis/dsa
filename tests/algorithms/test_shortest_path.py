from __future__ import annotations

import pytest
from pytest import approx

from dsa.algorithms import shortest_path
from dsa.data_structures import graphs


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_two_nodes(algorithm: shortest_path.ShortestPathAlgorithm, directed: bool) -> None:
    # node_1 -- 2.0 -- node_2
    graph = graphs.Graph(directed=directed)
    graph.add_value("node_1")
    graph.add_value("node_2")
    graph.add_connection("node_1", "node_2", weight=2.0)

    path_and_distance = algorithm(graph, source="node_1", target="node_2")
    assert path_and_distance.path == ["node_1", "node_2"]
    assert path_and_distance.distance == approx(2.0)

    if directed:
        with pytest.raises(AssertionError, match="Source and target are not connected"):
            path_and_distance = algorithm(graph, source="node_2", target="node_1")
    else:
        path_and_distance = algorithm(graph, source="node_2", target="node_1")
        assert path_and_distance.path == ["node_2", "node_1"]
        assert path_and_distance.distance == approx(2.0)


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_4_nodes(algorithm: shortest_path.ShortestPathAlgorithm, directed: bool) -> None:
    #             node_2
    #           /        \
    #         1.0        1.5
    #         /            \
    # node_1                node_4
    #         \            /
    #         2.0        1.0
    #           \        /
    #             node_3
    graph = graphs.Graph(directed=directed)
    graph.add_value("node_1")
    graph.add_value("node_2")
    graph.add_value("node_3")
    graph.add_value("node_4")
    graph.add_connection("node_1", "node_2", weight=1.0)
    graph.add_connection("node_2", "node_4", weight=1.5)
    graph.add_connection("node_1", "node_3", weight=2.0)
    graph.add_connection("node_3", "node_4", weight=1.0)

    path_and_distance = algorithm(graph, source="node_1", target="node_4")
    assert path_and_distance.path == ["node_1", "node_2", "node_4"]
    assert path_and_distance.distance == approx(2.5)

    if not directed:
        path_and_distance = algorithm(graph, source="node_3", target="node_2")
        assert path_and_distance.path == ["node_3", "node_4", "node_2"]
        assert path_and_distance.distance == approx(2.5)


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
@pytest.mark.parametrize("algorithm", [shortest_path.a_star_search_algorithm], ids=lambda f: f.__name__)
def test_shortest_path_7_nodes(algorithm: shortest_path.ShortestPathAlgorithm, directed: bool) -> None:
    #            node_3             node_5
    #          /        \          /      \
    #        6.0        8.0      10.0     2.0
    #        /            \      /           \
    # node_1               node_4           node_7
    #        \            /      \           /
    #        2.0        5.0      15.0      6.0
    #          \        /          \       /
    #            node_2             node_6
    graph = graphs.Graph(directed=directed)
    graph.add_value("node_1")
    graph.add_value("node_2")
    graph.add_value("node_3")
    graph.add_value("node_4")
    graph.add_value("node_5")
    graph.add_value("node_6")
    graph.add_value("node_7")
    graph.add_connection("node_1", "node_2", weight=2.0)
    graph.add_connection("node_1", "node_3", weight=6.0)
    graph.add_connection("node_2", "node_4", weight=5.0)
    graph.add_connection("node_3", "node_4", weight=8.0)
    graph.add_connection("node_4", "node_5", weight=10.0)
    graph.add_connection("node_4", "node_6", weight=15.0)
    graph.add_connection("node_5", "node_7", weight=2.0)
    graph.add_connection("node_6", "node_7", weight=6.0)

    path_and_distance = algorithm(graph, source="node_1", target="node_7")
    assert path_and_distance.path == ["node_1", "node_2", "node_4", "node_5", "node_7"]
    assert path_and_distance.distance == approx(19.0)

    if directed:
        path_and_distance = algorithm(graph, source="node_3", target="node_7")
        assert path_and_distance.path == ["node_3", "node_4", "node_5", "node_7"]
        assert path_and_distance.distance == approx(20.0)
    else:
        path_and_distance = algorithm(graph, source="node_7", target="node_2")
        assert path_and_distance.path == ["node_7", "node_5", "node_4", "node_2"]
        assert path_and_distance.distance == approx(17.0)
