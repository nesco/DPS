"""
Inclusion DAG (Hasse diagram) construction.

A Hasse diagram is the minimal DAG representing a partial order.
For set inclusion: an edge A → B means "B is directly contained in A"
(B ⊂ A with no C such that B ⊂ C ⊂ A).

This is AIT-general: works with any objects having a subset relation,
regardless of element type or how connectivity was defined.
"""

from collections.abc import Mapping, Set
from typing import Callable, TypeVar

from utils.dag_functionals import topological_sort

Element = TypeVar("Element")
Object = TypeVar("Object")


def build_hasse_diagram(
    objects: tuple[Object, ...],
    get_elements: Callable[[Object], frozenset[Element]],
) -> dict[Object, set[Object]]:
    """
    Build the Hasse diagram for subset inclusion among objects.

    For each object, finds its "direct children": objects that are
    immediately contained with no intermediate objects between them.

    Algorithm (O(n² × k) where n=objects, k=avg elements):
        1. Sort objects by size (largest first)
        2. For each object, scan smaller objects for direct children
        3. A child is "direct" if no other child contains it

    Args:
        objects: Objects to organize into a hierarchy.
        get_elements: Extracts the element set defining inclusion.
                     Object A contains B iff get_elements(B) ⊆ get_elements(A).

    Returns:
        DAG where graph[parent] = {direct children of parent}.
    """
    if not objects:
        return {}

    elements_of: dict[Object, frozenset[Element]] = {
        obj: get_elements(obj) for obj in objects
    }

    largest_first = sorted(objects, key=lambda obj: len(elements_of[obj]), reverse=True)

    graph: dict[Object, set[Object]] = {obj: set() for obj in objects}

    for i, parent in enumerate(largest_first):
        parent_elements = elements_of[parent]
        direct_children: list[Object] = []

        # Only smaller objects (later in sorted order) can be children
        for candidate in largest_first[i + 1 :]:
            candidate_elements = elements_of[candidate]

            is_proper_subset = candidate_elements < parent_elements
            if not is_proper_subset:
                continue

            # Direct child = not contained in any already-found child
            is_covered_by_existing_child = any(
                candidate_elements < elements_of[child] for child in direct_children
            )

            if not is_covered_by_existing_child:
                direct_children.append(candidate)

        graph[parent] = set(direct_children)

    return graph


def sort_by_inclusion(dag: Mapping[Object, Set[Object]]) -> tuple[Object, ...]:
    """
    Topologically sort objects so children come before parents.

    Useful for bottom-up processing: process contained objects
    before the objects that contain them.
    """
    return topological_sort(dag)
