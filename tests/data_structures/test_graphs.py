from __future__ import annotations

import pytest

from dsa.data_structures import graphs


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_add_value(directed: bool, representation: str) -> None:
    graph = graphs.Graph(representation=representation, directed=directed)
    assert not graph.has_value("1")
    assert not graph.has_value("2")

    graph.add_value("1")
    assert graph.has_value("1")
    assert not graph.has_value("2")

    graph.add_value("2")
    assert graph.has_value("1")
    assert graph.has_value("2")

    with pytest.raises(AssertionError):
        graph.add_value("1")


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_remove_value(directed: bool, representation: str) -> None:
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    assert graph.has_value("1")
    assert graph.has_value("2")

    graph.remove_value("1")
    assert not graph.has_value("1")
    assert graph.has_value("2")

    graph.remove_value("2")
    assert not graph.has_value("1")
    assert not graph.has_value("2")

    with pytest.raises(AssertionError):
        graph.remove_value("1")


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_add_connection(directed: bool, representation: str) -> None:
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    assert not graph.has_connection("1", "2")
    assert not graph.has_connection("2", "1")

    graph.add_connection("1", "2")
    assert graph.has_connection("1", "2")
    if directed:
        assert not graph.has_connection("2", "1")
    else:
        assert graph.has_connection("2", "1")

    with pytest.raises(AssertionError):
        graph.add_connection("1", "2")
    with pytest.raises(AssertionError):
        graph.add_connection("1", "3")


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_remove_connection(directed: bool, representation: str) -> None:
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    graph.add_connection("1", "2")
    assert graph.has_connection("1", "2")
    if directed:
        assert not graph.has_connection("2", "1")
    else:
        assert graph.has_connection("2", "1")

    graph.remove_connection("1", "2")
    assert not graph.has_connection("1", "2")
    assert not graph.has_connection("2", "1")

    with pytest.raises(AssertionError):
        graph.remove_connection("1", "2")


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_iterate_neighbors(directed: bool, representation: str) -> None:
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    graph.add_value("3")
    graph.add_connection("1", "2")
    graph.add_connection("2", "3")

    assert list(graph.iterate_neighbors("1")) == [("2", 1.0)]
    if directed:
        list(graph.iterate_neighbors("2")) == [("3", 1.0)]
        list(graph.iterate_neighbors("3")) == []
    else:
        list(graph.iterate_neighbors("2")) == [("1", 1.0), ("2", 1.0)]
        list(graph.iterate_neighbors("3")) == [("2", 1.0)]


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_traverse_BFS(directed: bool, representation: str) -> None:
    #   1
    #  / \
    # 2   3
    #    / \
    #   4 - 5
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    graph.add_value("3")
    graph.add_value("4")
    graph.add_value("5")
    graph.add_connection("1", "2")
    graph.add_connection("1", "3")
    graph.add_connection("3", "4")
    graph.add_connection("3", "5")
    graph.add_connection("4", "5")  # Add cycle

    assert list(graph.traverse_BFS("1")) == ["1", "2", "3", "4", "5"]
    if directed:
        assert list(graph.traverse_BFS("2")) == ["2"]
        assert list(graph.traverse_BFS("3")) == ["3", "4", "5"]
        assert list(graph.traverse_BFS("4")) == ["4", "5"]
        assert list(graph.traverse_BFS("5")) == ["5"]
    else:
        assert list(graph.traverse_BFS("2")) == ["2", "1", "3", "4", "5"]
        assert list(graph.traverse_BFS("3")) == ["3", "1", "4", "5", "2"]
        assert list(graph.traverse_BFS("4")) == ["4", "3", "5", "1", "2"]
        assert list(graph.traverse_BFS("5")) == ["5", "3", "4", "1", "2"]


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
@pytest.mark.parametrize("representation", ["adjacency_list", "adjacency_matrix", "pointers_and_objects"])
def test_graph_traverse_DFS(directed: bool, representation: str) -> None:
    #   1
    #  / \
    # 2   3
    #    / \
    #   4 - 5
    #  /
    # 6
    graph = graphs.Graph(representation=representation, directed=directed)
    graph.add_value("1")
    graph.add_value("2")
    graph.add_value("3")
    graph.add_value("4")
    graph.add_value("5")
    graph.add_value("6")
    graph.add_connection("1", "2")
    graph.add_connection("1", "3")
    graph.add_connection("3", "4")
    graph.add_connection("3", "5")
    graph.add_connection("4", "6")
    graph.add_connection("4", "5")  # Add cycle

    assert list(graph.traverse_DFS("1")) == ["1", "2", "3", "4", "5", "6"]
    if directed:
        assert list(graph.traverse_DFS("2")) == ["2"]
        assert list(graph.traverse_DFS("3")) == ["3", "4", "5", "6"]
        assert list(graph.traverse_DFS("4")) == ["4", "5", "6"]
        assert list(graph.traverse_DFS("5")) == ["5"]
    else:
        assert list(graph.traverse_DFS("2")) == ["2", "1", "3", "4", "5", "6"]
        assert list(graph.traverse_DFS("3")) == ["3", "1", "2", "4", "5", "6"]
        assert list(graph.traverse_DFS("4")) == ["4", "3", "1", "2", "5", "6"]
        assert list(graph.traverse_DFS("5")) == ["5", "3", "1", "2", "4", "6"]
