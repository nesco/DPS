"""
Functions for working with sets.
"""

import itertools
from collections.abc import Set
from typing import TypeVar, Callable

T = TypeVar("T")


def set_product(*sets: Set[T]) -> frozenset[tuple[T, ...]]:
    """
    Create a frozenset of tuples containing all possible combinations of elements,
    one from each input set (Cartesian product).
    """
    if not sets:
        return frozenset()

    return frozenset(itertools.product(*sets))
