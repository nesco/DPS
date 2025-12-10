"""
Component extraction from grids.

Extracts connected components from grids using 8-connectivity (King moves).
Components are grouped by their color sets.

This module is ARC-specific: it uses position-based connectivity (King moves)
and groups components by color attributes. The underlying decomposition
primitives are in the decomposition/ package.
"""

from collections import defaultdict
from itertools import chain, combinations

from decomposition import extract_connected_components, king_neighbors
from localtypes import (
    Color,
    ColorGrid,
    Colors,
    Coord,
    Coords,
)
from utils.grid import grid_to_coords_by_color


def coords_to_connected_components(coords: Coords) -> frozenset[Coords]:
    """
    Find connected components in a set of coordinates using 8-connectivity.

    Uses King moves (8-connectivity) from the decomposition package.

    Args:
        coords: Set of coordinates to partition.

    Returns:
        Frozenset of coordinate sets, each representing a connected component.
    """
    return extract_connected_components(frozenset(coords), king_neighbors)


def condition_by_color_couples(
    grid: ColorGrid,
) -> dict[tuple[Color, Color], Coords]:
    """
    Returns a dict of all masks comprised of union of two colours.

    Note: This is legacy code, consider using condition_by_colors instead.
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = sorted(list(coords_by_color.keys()))

    coords_by_color_couple: dict[tuple[Color, Color], Coords] = {}

    for i in range(len(colors)):
        color_a = colors[i]
        for j in range(i):
            color_b = colors[j]
            coords_by_color_couple[(color_a, color_b)] = (
                coords_by_color[color_a] | coords_by_color[color_b]
            )

        coords_by_color_couple[(color_a, color_a)] = coords_by_color[color_a]
    return coords_by_color_couple


def condition_by_colors(grid: ColorGrid) -> dict[Colors, Coords]:
    """
    Get coordinate masks for all color subsets.

    Returns a dictionary where each key is a frozenset of colors (representing a subset)
    and each value is the set of coordinates where the pixel color is in that subset.

    Args:
        grid: Input color grid.

    Returns:
        Dictionary mapping color subsets to their coordinate masks.
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = list(coords_by_color.keys())
    all_subsets = chain.from_iterable(
        combinations(colors, r) for r in range(1, len(colors) + 1)
    )
    result = {}
    for subset in all_subsets:
        S = frozenset(subset)
        union_coords = set().union(*(coords_by_color[c] for c in S))
        result[S] = union_coords
    return result


def grid_to_components_by_colors(
    grid: ColorGrid,
) -> defaultdict[Colors, set[Coords]]:
    """
    Extract connected components from a grid, grouped by their color sets.

    For each color subset, finds connected components in the union of those colors.
    Components are deduplicated: each component is associated with its minimal color set.

    Args:
        grid: Input color grid.

    Returns:
        Dictionary mapping color sets to their connected components.
    """
    components_by_colors = {}
    coords_by_colors = condition_by_colors(grid)
    for colors, coords in coords_by_colors.items():
        components_by_colors[colors] = coords_to_connected_components(coords)

    # Removing multiple occurrences of components associated with several color sets
    # A single colored component of color 1 like the cross of "2dc579da.json"
    # will appear in the coords mask of {1} and {1, 3} for example
    colors_by_component = {}
    for colors, components in components_by_colors.items():
        for component in components:
            if component not in colors_by_component:
                colors_by_component[component] = colors
            else:
                if len(colors) < len(colors_by_component[component]):
                    # Keep the minimal color set
                    colors_by_component[component] = colors

    # Invert back: group components by their minimal color sets
    new_components_by_colors: defaultdict[Colors, set] = defaultdict(set)
    for component, colors in colors_by_component.items():
        new_components_by_colors[colors].add(component)

    return new_components_by_colors
