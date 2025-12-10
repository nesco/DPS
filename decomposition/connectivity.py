"""
Connectivity definitions for decomposition.

Connectivity defines adjacency between elements, enabling connected component
extraction. This module reuses the existing freeman direction definitions
rather than duplicating them.

The key insight: an Alphabet (like MoveValue) defines symbols, and the
corresponding DIRECTIONS define the deltas. Connectivity is the adjacency
relation derived from those deltas.
"""

from collections.abc import Sequence
from typing import Callable, TypeVar

from freeman import BISHOP, DIRECTIONS_FREEMAN, KING, TOWER, King
from localtypes import Coord, Coords

# Type for a neighbor function
E = TypeVar("E")
NeighborFunc = Callable[[E, frozenset[E]], frozenset[E]]


def make_coord_neighbors(
    directions: Sequence[King],
) -> Callable[[Coord, frozenset[Coord]], frozenset[Coord]]:
    """
    Create a neighbor function for coordinates using given directions.

    Args:
        directions: List of direction indices (from freeman.py).

    Returns:
        Function that computes neighbors of a coordinate within a universe.
    """
    deltas = tuple(DIRECTIONS_FREEMAN[d] for d in directions)

    def neighbors(coord: Coord, universe: frozenset[Coord]) -> frozenset[Coord]:
        col, row = coord
        candidates = frozenset(
            Coord(col + dcol, row + drow) for dcol, drow in deltas
        )
        return candidates & universe

    return neighbors


# Pre-built neighbor functions for common connectivities
king_neighbors = make_coord_neighbors(KING)  # 8-connectivity
tower_neighbors = make_coord_neighbors(TOWER)  # 4-connectivity (orthogonal)
bishop_neighbors = make_coord_neighbors(BISHOP)  # 4-connectivity (diagonal)
