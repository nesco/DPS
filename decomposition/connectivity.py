"""
Connectivity definitions for decomposition.

A connectivity defines which elements are "neighbors" of each other,
enabling connected component extraction. Different connectivities
produce different decompositions of the same data.

This module builds on freeman.py's direction definitions:
- KING: 8 directions (orthogonal + diagonal) → 8-connectivity
- TOWER: 4 orthogonal directions → 4-connectivity
- BISHOP: 4 diagonal directions → diagonal-only connectivity
"""

from collections.abc import Sequence
from typing import Callable, TypeVar

from freeman import BISHOP, DIRECTIONS_FREEMAN, KING, TOWER, King
from localtypes import Coord

E = TypeVar("E")

# A neighbor function takes an element and a universe, returns neighbors within that universe
NeighborFunc = Callable[[E, frozenset[E]], frozenset[E]]

# Specific type for coordinate-based neighbor functions
CoordNeighborFunc = Callable[[Coord, frozenset[Coord]], frozenset[Coord]]


def make_coord_neighbors(directions: Sequence[King]) -> CoordNeighborFunc:
    """
    Create a neighbor function from a set of movement directions.

    The returned function computes which coordinates are reachable
    from a given coordinate by moving one step in any of the specified directions.

    Args:
        directions: Movement directions (indices into DIRECTIONS_FREEMAN).
                   Use KING for 8-connectivity, TOWER for 4-connectivity.

    Returns:
        A function (coord, universe) -> neighbors within universe.

    Example:
        >>> neighbors = make_coord_neighbors(KING)
        >>> neighbors(Coord(1, 1), frozenset([Coord(0, 0), Coord(1, 0), Coord(2, 2)]))
        frozenset({Coord(col=0, row=0), Coord(col=2, row=2)})
    """
    deltas: tuple[Coord, ...] = tuple(DIRECTIONS_FREEMAN[d] for d in directions)

    def neighbors(coord: Coord, universe: frozenset[Coord]) -> frozenset[Coord]:
        col, row = coord
        adjacent = frozenset(
            Coord(col + delta.col, row + delta.row) for delta in deltas
        )
        return adjacent & universe

    return neighbors


# Standard connectivity functions for 2D grids
king_neighbors: CoordNeighborFunc = make_coord_neighbors(KING)
tower_neighbors: CoordNeighborFunc = make_coord_neighbors(TOWER)
bishop_neighbors: CoordNeighborFunc = make_coord_neighbors(BISHOP)
