"""
Generic object extraction via connected components.

Objects are connected sets of elements. What constitutes "connected"
is determined by the neighbor function (connectivity).

This module provides AIT-general object extraction that can work with
any element type and connectivity definition.
"""

from typing import Callable, TypeVar

from utils.graph import nodes_to_connected_components

E = TypeVar("E")  # Element type


def extract_connected_components(
    elements: frozenset[E],
    neighbors: Callable[[E, frozenset[E]], frozenset[E]],
) -> frozenset[frozenset[E]]:
    """
    Extract connected components from elements using given connectivity.

    This is the core AIT-general decomposition primitive. The specific
    notion of "connected" is defined by the neighbors function.

    Args:
        elements: Set of elements to partition.
        neighbors: Function returning neighbors of an element within a universe.

    Returns:
        Frozenset of connected components (each a frozenset of elements).
    """
    if not elements:
        return frozenset()

    def node_to_neighbours(node: E) -> frozenset[E]:
        return neighbors(node, elements)

    return nodes_to_connected_components(elements, node_to_neighbours)
