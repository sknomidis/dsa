from __future__ import annotations

import numpy as np
import pytest

from dsa.data_structures import trees


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_insert(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    for _ in range(4):
        values = sorted(np.unique(rng.integers(0, 100, 42)))
        heap = heap_type()
        for element in rng.permuted(values).tolist():
            heap.insert(element)
        assert heap.get_extremum() == values[0 if heap_type == trees.MinHeap else -1]


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_pop(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    values = sorted(np.unique(rng.integers(0, 100, 42)))
    heap = heap_type()
    for value in rng.permuted(values).tolist():
        heap.insert(value)
    for _ in range(len(values)):
        assert heap.get_extremum() == values[0 if heap_type == trees.MinHeap else -1]
        assert heap.pop() == (values.pop(0) if heap_type == trees.MinHeap else values.pop())
    assert heap.size == 0


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_delete(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    values = list(range(100))
    heap = heap_type()
    for value in rng.permuted(values).tolist():
        heap.insert(value)

    for value in rng.permuted(list(values)).tolist():
        values.remove(value)
        heap.delete(value)
        if values:
            assert heap.get_extremum() == min(values) if heap_type == trees.MinHeap else max(values)
