from __future__ import annotations

from typing import Sequence

import pytest

from dsa.data_structures import stacks


@pytest.mark.parametrize("elements", [[0], [1, 2], [1, 2, 4], [1, 2, 4, 8]], ids=lambda s: f"size-{len(s)}")
@pytest.mark.parametrize(
    "stack_type",
    [stacks.ArrayStack, stacks.ListStack],
    ids=lambda c: c.__name__,
)
def test_stack(elements: Sequence[int], stack_type: type[stacks.Stack]) -> None:
    stack = stack_type()
    for element in elements:
        stack.push(element)
        assert stack.pop() == element
        stack.push(element)
    assert [stack.pop() for _ in range(len(elements))] == elements[::-1]
    with pytest.raises(AssertionError):
        stack.pop()
