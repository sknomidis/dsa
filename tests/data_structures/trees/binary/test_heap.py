from __future__ import annotations

import numpy as np
import pytest

from dsa.data_structures import trees


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_insert(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    for _ in range(4):
        elements = sorted(np.unique(rng.integers(0, 100, 42)))
        heap = heap_type()
        for element in rng.permuted(elements).tolist():
            heap.insert(element)
        assert heap.get_extremum() == elements[0 if heap_type == trees.MinHeap else -1]


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_extract_extremum(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    elements = sorted(np.unique(rng.integers(0, 100, 42)))
    heap = heap_type(rng.permuted(elements).tolist())
    for _ in range(len(elements)):
        assert heap.get_extremum() == elements[0 if heap_type == trees.MinHeap else -1]
        assert heap.extract_extremum() == (elements.pop(0) if heap_type == trees.MinHeap else elements.pop())
    assert heap.size == 0


@pytest.mark.parametrize("heap_type", [trees.MinHeap, trees.MaxHeap], ids=lambda c: c.__name__)
def test_heap_delete(heap_type: type[trees.Heap], rng: np.random.Generator) -> None:
    elements = sorted(np.unique(rng.integers(0, 100, 42)))
    heap = heap_type(elements)
    while elements:
        assert heap.get_extremum() == elements[0 if heap_type == trees.MinHeap else -1]
        element = rng.choice(elements)
        heap.delete(element)
        elements.remove(element)
        with pytest.raises(AssertionError):
            heap.delete(element)
    assert heap.is_empty
