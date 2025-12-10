"""
DAG (Directed Acyclic Graph) utilities.

Functions:
    topological_sort(graph) - Kahn's algorithm for topological ordering
"""

from collections import deque
from collections.abc import Mapping, Set
from typing import TypeVar

T = TypeVar("T")


def topological_sort(parent_to_children: Mapping[T, Set[T]]) -> tuple[T, ...]:
    """
    Returns nodes in topological order using Kahn's algorithm.

    Args:
        parent_to_children: Graph as adjacency list (node -> set of dependents)

    Returns:
        Nodes ordered so parents come before children.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    all_nodes: set[T] = set(parent_to_children.keys())
    for children in parent_to_children.values():
        all_nodes.update(children)

    in_degree: dict[T, int] = {node: 0 for node in all_nodes}
    for parent, children in parent_to_children.items():
        for child in children:
            in_degree[child] += 1

    queue = deque(node for node in all_nodes if in_degree[node] == 0)
    sorted_list: list[T] = []

    while queue:
        node = queue.popleft()
        sorted_list.append(node)

        for child in parent_to_children.get(node, ()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(sorted_list) != len(all_nodes):
        raise ValueError("Graph contains a cycle")

    return tuple(sorted_list)
