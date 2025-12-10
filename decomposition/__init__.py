"""
AIT-general decomposition primitives.

This package provides domain-agnostic primitives for decomposing structured data:

**Connectivity** (connectivity.py)
    Defines adjacency relations. Different connectivities produce different
    decompositions of the same data.
    - king_neighbors: 8-connectivity (orthogonal + diagonal)
    - tower_neighbors: 4-connectivity (orthogonal only)
    - bishop_neighbors: diagonal-only connectivity

**Objects** (objects.py)
    Connected component extraction parameterized by connectivity.
    - extract_connected_components(elements, neighbors) -> components

**Inclusion** (inclusion.py)
    Hasse diagram (DAG) construction for subset relationships.
    - build_hasse_diagram(objects, get_elements) -> DAG

The ARC-specific instantiation (grids, colors, pixels) lives in hierarchy/.
"""

from .connectivity import (
    CoordNeighborFunc,
    NeighborFunc,
    bishop_neighbors,
    king_neighbors,
    make_coord_neighbors,
    tower_neighbors,
)
from .inclusion import (
    build_hasse_diagram,
    sort_by_inclusion,
)
from .objects import (
    extract_connected_components,
)

__all__ = [
    # Connectivity
    "NeighborFunc",
    "CoordNeighborFunc",
    "make_coord_neighbors",
    "king_neighbors",
    "tower_neighbors",
    "bishop_neighbors",
    # Objects
    "extract_connected_components",
    # Inclusion
    "build_hasse_diagram",
    "sort_by_inclusion",
]
