from __future__ import annotations

from typing import Sequence

import pytest

from dsa.data_structures import queues


@pytest.mark.parametrize("elements", [[0], [1, 2], [1, 2, 4], [1, 2, 4, 8]], ids=lambda s: f"size-{len(s)}")
@pytest.mark.parametrize(
    "queue_type",
    [
        queues.ArrayQueue,
        queues.ListQueue,
    ],
    ids=lambda c: c.__name__,
)
def test_queue(elements: Sequence[int], queue_type: type[queues.Queue]) -> None:
    queue = queue_type()
    for element in elements:
        queue.enqueue(element)
    assert [queue.dequeue() for _ in range(len(elements))] == elements
    with pytest.raises(AssertionError):
        queue.dequeue()

    for element in elements:
        queue.enqueue(element)
    for _ in range(len(elements)):
        element = queue.dequeue()
        queue.enqueue(element)
    assert [queue.dequeue() for _ in range(len(elements))] == elements
