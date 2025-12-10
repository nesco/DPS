"""
Inclusion DAG (Hasse diagram) construction.

Builds a DAG representing the inclusion relation between objects.
Each edge represents direct inclusion (no intermediate objects).

This is AIT-general - it works with any objects that have a subset relation,
regardless of what the elements are or how connectivity was defined.
"""

from collections.abc import Mapping, Set
from typing import Callable, TypeVar

from utils.dag_functionals import topological_sort

E = TypeVar("E")  # Element type
O = TypeVar("O")  # Object type (typically has elements + attributes)


def build_hasse_diagram(
    objects: tuple[O, ...],
    get_elements: Callable[[O], frozenset[E]],
) -> dict[O, set[O]]:
    """
    Build the inclusion DAG (Hasse diagram) from objects.

    Constructs a DAG where:
    - Nodes are objects
    - Edges represent direct inclusion (A -> B means B is directly contained in A)

    Args:
        objects: Tuple of objects to organize.
        get_elements: Function to extract the element set from an object.

    Returns:
        DAG mapping each object to its direct children (included objects).
    """
    # Build subsets and supersets dictionaries
    subsets = {
        a: frozenset(b for b in objects if get_elements(b) <= get_elements(a))
        for a in objects
    }
    supersets = {
        b: frozenset(c for c in objects if get_elements(b) <= get_elements(c))
        for b in objects
    }

    # Initialize the DAG
    graph: dict[O, set[O]] = {a: set() for a in objects}

    # For each object a, find direct children (proper subsets with no intermediate)
    for a in objects:
        for b in subsets[a]:
            if b != a:  # Ensure b is a proper subset of a
                intersection = subsets[a] & supersets[b]
                if len(intersection) == 2:  # Only a and b, no intermediate c
                    graph[a].add(b)

    return graph


def sort_by_inclusion(dag: Mapping[O, Set[O]]) -> tuple[O, ...]:
    """
    Topologically sort objects by inclusion.

    Returns objects sorted such that included objects come before their containers.

    Args:
        dag: The inclusion DAG.

    Returns:
        Tuple of objects in topological order.
    """
    return topological_sort(dag)
