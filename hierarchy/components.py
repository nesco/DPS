"""
Component extraction from ARC grids.

Extracts connected components grouped by their color sets using 8-connectivity.

Key concepts:
- **Atom**: A single-color connected component (the smallest unit)
- **Component**: A multi-color connected region (union of adjacent atoms)
- **Minimal color set**: The smallest set of colors that produces a given component

Algorithm overview:
    Old approach: O(2^c × pixels)
        For each of 2^c color subsets, scan all pixels to find components.

    New approach: O(pixels + 2^c × atoms)
        1. Find atoms (single-color components) once - O(pixels)
        2. Build adjacency graph between atoms - O(pixels)
        3. For each color subset, find components in atom graph - O(atoms)

    Since atoms << pixels typically, this is much faster.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations

from decomposition import extract_connected_components, king_neighbors
from arc.types import (
    Color,
    ColorGrid,
    Colors,
    Coord,
    Coords,
)
from utils.graph import nodes_to_connected_components
from utils.grid import grid_to_coords_by_color


@dataclass(frozen=True)
class Atom:
    """
    A single-color connected component.

    Atoms are the building blocks for multi-color components.
    Two atoms can merge into a larger component if they are adjacent
    and both their colors are in the target color set.
    """

    color: Color
    coords: frozenset[Coord]


# Type aliases for clarity
AtomGraph = dict[Atom, set[Atom]]
CoordToAtom = dict[Coord, Atom]


def coords_to_connected_components(coords: Coords) -> frozenset[Coords]:
    """Partition coordinates into 8-connected components."""
    return extract_connected_components(frozenset(coords), king_neighbors)


def condition_by_color_couples(
    grid: ColorGrid,
) -> dict[tuple[Color, Color], Coords]:
    """
    Get coordinate masks for all pairs of colors.

    Deprecated: Use grid_to_components_by_colors instead.
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

    For each non-empty subset S of colors in the grid, returns the set of
    coordinates where the pixel color is in S.

    Example:
        Grid with colors {0, 1, 2} produces 7 masks:
        {0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = list(coords_by_color.keys())
    all_subsets = chain.from_iterable(
        combinations(colors, r) for r in range(1, len(colors) + 1)
    )
    return {
        frozenset(subset): set().union(*(coords_by_color[c] for c in subset))
        for subset in all_subsets
    }


def _find_atoms(grid: ColorGrid) -> tuple[list[Atom], CoordToAtom]:
    """
    Extract all atoms (single-color connected components) from a grid.

    Returns:
        atoms: All atoms in the grid.
        coord_to_atom: Maps each coordinate to the atom containing it.
    """
    coords_by_color = grid_to_coords_by_color(grid)
    atoms: list[Atom] = []
    coord_to_atom: CoordToAtom = {}

    for color, color_coords in coords_by_color.items():
        for component in coords_to_connected_components(color_coords):
            atom = Atom(color, frozenset(component))
            atoms.append(atom)
            for coord in component:
                coord_to_atom[coord] = atom

    return atoms, coord_to_atom


def _build_atom_adjacency(atoms: list[Atom], coord_to_atom: CoordToAtom) -> AtomGraph:
    """
    Build the adjacency graph between atoms.

    Two atoms are adjacent if any pair of their pixels are 8-connected neighbors.
    This graph is used to find multi-color components efficiently.
    """
    all_coords = frozenset(coord_to_atom.keys())
    adjacency: AtomGraph = {atom: set() for atom in atoms}

    for atom in atoms:
        for coord in atom.coords:
            for neighbor_coord in king_neighbors(coord, all_coords):
                neighbor_atom = coord_to_atom[neighbor_coord]
                if neighbor_atom != atom:
                    adjacency[atom].add(neighbor_atom)

    return adjacency


def _merge_atom_coords(atoms: frozenset[Atom]) -> frozenset[Coord]:
    """Combine coordinates from multiple atoms into one set."""
    all_coords: set[Coord] = set()
    for atom in atoms:
        all_coords.update(atom.coords)
    return frozenset(all_coords)


def _find_components_for_color_subset(
    color_set: Colors,
    atoms: list[Atom],
    atom_adjacency: AtomGraph,
) -> frozenset[frozenset[Coord]]:
    """
    Find connected components for a specific color subset.

    Filters atoms to those matching the color set, then finds
    connected components in the restricted atom adjacency graph.
    """
    matching_atoms = frozenset(atom for atom in atoms if atom.color in color_set)

    if not matching_atoms:
        return frozenset()

    def restricted_neighbors(atom: Atom) -> frozenset[Atom]:
        return frozenset(a for a in atom_adjacency[atom] if a in matching_atoms)

    atom_components = nodes_to_connected_components(
        matching_atoms, restricted_neighbors
    )

    return frozenset(_merge_atom_coords(ac) for ac in atom_components)


def _deduplicate_by_minimal_colors(
    components_by_colors: dict[Colors, frozenset[frozenset[Coord]]],
) -> defaultdict[Colors, set[Coords]]:
    """
    Associate each component with its minimal color set.

    A component may appear in multiple color subsets (e.g., a red component
    appears in {red}, {red, blue}, {red, green}, etc.). We keep only the
    minimal color set that produces each component.
    """
    minimal_colors_of: dict[frozenset[Coord], Colors] = {}

    for color_set, components in components_by_colors.items():
        for component in components:
            current_minimal = minimal_colors_of.get(component)
            if current_minimal is None or len(color_set) < len(current_minimal):
                minimal_colors_of[component] = color_set

    # Group components by their minimal color sets
    result: defaultdict[Colors, set[Coords]] = defaultdict(set)
    for component, color_set in minimal_colors_of.items():
        result[color_set].add(component)

    return result


def grid_to_components_by_colors(grid: ColorGrid) -> defaultdict[Colors, set[Coords]]:
    """
    Extract all connected components from a grid, grouped by color sets.

    Each component is associated with its minimal color set: the smallest
    subset of colors that produces that exact component.

    Example:
        A grid with an isolated red cross and a blue square would produce:
        - {red}: {cross_coords}
        - {blue}: {square_coords}
        - {red, blue}: (only if they're adjacent and merge)

    Returns:
        Mapping from minimal color sets to their components.
    """
    atoms, coord_to_atom = _find_atoms(grid)

    if not atoms:
        return defaultdict(set)

    atom_adjacency = _build_atom_adjacency(atoms, coord_to_atom)

    # Find components for each color subset
    all_colors = frozenset(atom.color for atom in atoms)
    all_color_subsets = chain.from_iterable(
        combinations(all_colors, r) for r in range(1, len(all_colors) + 1)
    )

    components_by_colors: dict[Colors, frozenset[frozenset[Coord]]] = {
        frozenset(subset): _find_components_for_color_subset(
            frozenset(subset), atoms, atom_adjacency
        )
        for subset in all_color_subsets
    }

    return _deduplicate_by_minimal_colors(components_by_colors)
