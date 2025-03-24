"""
Module for functions of Direct Acyclic Graphs
"""

from collections import deque
from typing import Any, TypeVar

from collections.abc import Mapping, Set

T = TypeVar('T')

def topological_sort(
    parent_to_children: Mapping[T, Set[T]],
) -> tuple[T, ...]:
    sorted_list = []

    # Step 1: Get all nodes
    all_nodes = set(parent_to_children.keys()) | {
        node for children in parent_to_children.values() for node in children
    }

    # Step 2: Compute in-degrees
    in_degrees = {node: 0 for node in all_nodes}
    for node in all_nodes:
        for child in parent_to_children[node]:
            in_degrees[child] += 1

    # Step 3: Initialize queue of nodes with 0 in degrees
    # It's an invariant of the algorithm'
    queue = deque([node for node in all_nodes if in_degrees[node] == 0])

    # Step 4: Process each node
    # Add them to the sorted list
    # Decrease the in degrees of its children
    # And append children with 0 in degrees left
    while queue:
        node = queue.popleft()
        sorted_list.append(node)
        for child in parent_to_children[node]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                queue.append(child)

    return tuple(sorted_list)
