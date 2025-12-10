"""
Functions related to graphs
"""

from collections import deque
from typing import Callable, Sequence, TypeVar

from collections.abc import Set

T = TypeVar("T")


def nodes_to_connected_components(
    nodes: Set[T], node_to_neighbours: Callable[[T], Set[T]]
) -> frozenset[frozenset[T]]:
    """
    Extract connected components from an undirected graph structure.

    Args:
        nodes: set of nodes in the graph.
        connected_to: Function returning the set of nodes a given node points to.

    Returns:
        frozenset[set[T]]: set of connected components of the graph
    """

    seen = set()
    components = set()

    # Guarantees all the nodes are at least visited once
    for node in nodes:
        # Avoid visinting an already seen component
        if node in seen:
            continue

        # Add a new connected component
        component = set()
        # Breadth-first traversal
        queue = deque([node])
        while queue:
            current = queue.popleft()

            # Avoid cycles
            if current in seen:
                continue

            component.add(current)
            queue.extend(node_to_neighbours(current))

            # Mark the node as seen
            seen.add(current)

        # Add the completed component
        components.add(frozenset(component))
    return frozenset(components)
