"""
DAG construction for grid object inclusion hierarchy.

Builds a Hasse diagram (minimal DAG) representing the inclusion relation
between grid objects. Each edge represents direct inclusion (no intermediate objects).

This module is ARC-specific: it works with GridObjects (coords + colors).
The underlying Hasse diagram algorithm is in decomposition/inclusion.py.
"""

from collections.abc import Mapping, Set

from decomposition import build_hasse_diagram
from decomposition import sort_by_inclusion as generic_sort_by_inclusion
from arc.types import Colors, Coords
from hierarchy.types import GridObject


def components_by_colors_to_grid_object_dag(
    components_by_colors: Mapping[Colors, Set[Coords]],
) -> dict[GridObject, set[GridObject]]:
    """
    Build the inclusion DAG (Hasse diagram) from components.

    Constructs a DAG where:
    - Nodes are GridObjects (components with their color sets)
    - Edges represent direct inclusion (A -> B means B is directly contained in A)

    The DAG forms a lattice-like structure where selecting all colors creates
    a single component encompassing the entire grid (the supremum).

    Uses the generic Hasse diagram algorithm from decomposition/.

    Args:
        components_by_colors: Sets of components grouped by their color sets.

    Returns:
        DAG mapping each GridObject to its direct children (included objects).
    """
    # Build GridObjects
    grid_objects = tuple(
        GridObject(colors, component)
        for colors, components in components_by_colors.items()
        for component in components
    )

    # Use generic Hasse diagram builder with coords as elements
    return build_hasse_diagram(grid_objects, lambda go: frozenset(go.coords))


def sort_by_inclusion(
    grid_object_dag: Mapping[GridObject, Set[GridObject]],
) -> tuple[GridObject, ...]:
    """
    Topologically sort grid objects by inclusion.

    Returns objects sorted such that included objects come before their containers.

    Args:
        grid_object_dag: The inclusion DAG.

    Returns:
        Tuple of GridObjects in topological order.
    """
    return generic_sort_by_inclusion(grid_object_dag)
