"""
Grid hierarchy and syntax tree construction.

This module converts ARC grids into hierarchical syntax trees:
1. Extract connected components by color combinations
2. Build inclusion DAG between components
3. Convert to syntax trees with background normalization

Main entry point: grid_to_syntax_trees()
"""

from kolmogorov_tree import (
    MoveValue,
    ProductNode,
    RootNode,
    SumNode,
)
from localtypes import ColorGrid, GridObject

from .components import (
    condition_by_color_couples,
    condition_by_colors,
    coords_to_connected_components,
    grid_to_components_by_colors,
)
from .dag import (
    components_by_colors_to_grid_object_dag,
    sort_by_inclusion,
)
from .syntax import (
    dag_to_syntax_trees,
    dag_to_syntax_trees_linear,
    unpack_dependencies,
)


def grid_to_syntax_trees(
    grid: ColorGrid,
) -> tuple[
    dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ],
    tuple[GridObject, ...],
]:
    """
    Convert a grid to hierarchical syntax trees.

    Pipeline:
    1. grid_to_components_by_colors: Extract connected components by colors
    2. components_by_colors_to_grid_object_dag: Build inclusion DAG
    3. dag_to_syntax_trees: Convert to syntax trees with normalization

    Args:
        grid: A ColorGrid representing the input grid.

    Returns:
        Tuple containing:
        - Dictionary mapping GridObjects to their syntax trees
        - Tuple of GridObjects in topological order (included first)
    """
    # Step 1: Convert grid to components by colors
    components_by_colors = grid_to_components_by_colors(grid)

    # Step 2: Convert components to inclusion DAG
    grid_object_dag = components_by_colors_to_grid_object_dag(
        components_by_colors
    )

    # Step 3: Convert DAG to syntax trees
    syntax_trees, sorted_grid_objects = dag_to_syntax_trees(grid_object_dag)

    return syntax_trees, sorted_grid_objects


__all__ = [
    # Main entry point
    "grid_to_syntax_trees",
    # Components
    "coords_to_connected_components",
    "condition_by_colors",
    "condition_by_color_couples",
    "grid_to_components_by_colors",
    # DAG
    "components_by_colors_to_grid_object_dag",
    "sort_by_inclusion",
    # Syntax
    "dag_to_syntax_trees",
    "dag_to_syntax_trees_linear",
    "unpack_dependencies",
]
