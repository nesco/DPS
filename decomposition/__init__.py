"""
AIT-general decomposition primitives.

This package provides algorithmic information theory-grounded primitives
for decomposing structured data:

1. **Connectivity**: Defines adjacency relations (reuses freeman directions)
2. **Objects**: Connected component extraction parameterized by connectivity
3. **Inclusion**: Hasse diagram (DAG) construction for subset relationships

These primitives are domain-agnostic. The ARC-specific instantiation
(position-based objects with color attributes) lives in the hierarchy/ package.

Key insight: The same decomposition algorithm can use different connectivities
(King moves for position, transitions for color) to define different notions
of "object". MDL can select the best decomposition.
"""

from .connectivity import (
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
