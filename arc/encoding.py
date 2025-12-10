"""
ARC grid encoding to Kolmogorov syntax trees.

Transforms connected components from pixel coordinates into abstract
syntax trees (KNodes) using Freeman chain codes as intermediate representation.

Encoding pipeline:
    Coords → FreemanNode → KNode → RootNode

Key functions:
    - syntax_tree_at: Encode a component from a specific starting point
    - encode_component: Generate distribution of encodings with symbolization
    - encode_component_distribution: Generate raw distribution without symbolization
"""

from dataclasses import dataclass
from typing import Generic, cast

from freeman import (
    FreemanNode,
    TraversalModes,
    encode_connected_component,
)
from kolmogorov_tree import (
    CoordValue,
    CountValue,
    KNode,
    MoveValue,
    NoneValue,
    PaletteValue,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RootNode,
    T,
    expand_repeats,
    extract_nested_patterns,
    factorize_tuple,
    iterable_to_product,
    iterable_to_sum,
    postmap,
    premap,
    symbolize,
)
from arc.types import (
    Colors,
    Coord,
    Coords,
)


# --- Rectangle Detection ---


def detect_rectangle_pattern(moves: str) -> tuple[int, int] | None:
    """
    Detect if a move sequence forms a filled rectangle.

    The rectangle is assumed to be traversed DFS-style from top-left corner:
    right moves (2), then down (3), alternating left/right per row.

    Args:
        moves: String of Freeman chain code directions (0-7).

    Returns:
        (height, width) if rectangle detected, None otherwise.
    """
    if not moves or len(moves) <= 2:
        return None

    width = moves.index("3") + 1 if "3" in moves else len(moves) + 1
    height = moves.count("3") + 1

    # Expected pattern: "222...3" + alternating rows
    expected = "2" * (width - 1) + "".join(
        "3" + ("0" if i % 2 else "2") * (width - 1) for i in range(1, height)
    )

    if moves == expected and height >= 2 and width >= 2:
        return (height, width)
    return None


def extract_moves_from_product(knode: KNode) -> str | None:
    """Extract move string from a ProductNode of PrimitiveNode[MoveValue]."""
    if not isinstance(knode, ProductNode):
        return None

    node = cast(ProductNode, expand_repeats(knode))
    moves = []
    for child in node.children:
        if isinstance(child, PrimitiveNode) and isinstance(child.value, MoveValue):
            moves.append(str(child.data))
        else:
            return None
    return "".join(moves)


def apply_rectangle_detection(node: KNode) -> KNode:
    """
    Replace a ProductNode with RectNode if it encodes a rectangle.

    This is a tree transformation that detects rectangle patterns
    in move sequences and replaces them with the more compact RectNode.
    """
    moves_str = str(expand_repeats(node))
    if moves_str:
        rect = detect_rectangle_pattern(moves_str)
        if rect:
            height, width = rect
            return RectNode(CountValue(height), CountValue(width))
    return node


# --- Freeman to KNode Conversion ---


def freeman_to_knode(freeman: FreemanNode) -> KNode[MoveValue]:
    """
    Convert FreemanNode tree to KNode without compression.

    Translates the path-based Freeman representation into a product/sum
    structure of primitive move nodes.

    Args:
        freeman: Freeman chain code tree.

    Returns:
        KNode representing the same traversal.

    Raises:
        ValueError: If freeman has both empty path and no children.
    """
    product = []

    # Encode the main path as sequence of moves
    moves = [PrimitiveNode(MoveValue(move)) for move in freeman.path]
    if moves:
        product.extend(moves)

    # Recursively encode branching children as a SumNode
    children = iterable_to_sum(
        frozenset({freeman_to_knode(child) for child in freeman.children}), True
    )
    if children:
        product.append(children)

    result = iterable_to_product(product)

    if not result:
        raise ValueError("Freeman node has both empty path and no children")

    return result


def compress_knode(uncompressed: KNode[MoveValue]) -> KNode[MoveValue]:
    """
    Apply compression transformations to a KNode.

    Currently applies:
    - Factorization of repeated patterns
    - Rectangle pattern detection

    Args:
        uncompressed: Raw KNode from freeman_to_knode.

    Returns:
        Compressed KNode with repeated patterns factored out.
    """
    factored = postmap(uncompressed, factorize_tuple)
    with_rects = premap(factored, apply_rectangle_detection)
    return with_rects


def encode_freeman_to_knode(freeman: FreemanNode) -> KNode[MoveValue]:
    """
    Convert FreemanNode to compressed KNode.

    Combines conversion and compression into single function.
    """
    raw = freeman_to_knode(freeman)
    return compress_knode(raw)


# --- RootNode Construction ---


def create_root_node(
    freeman: FreemanNode,
    start: Coord,
    colors: Colors,
) -> RootNode:
    """
    Create RootNode from Freeman tree with metadata.

    Args:
        freeman: Freeman chain code tree for the component.
        start: Starting coordinate of the traversal.
        colors: Color palette of the component.

    Returns:
        RootNode wrapping the encoded tree with position and color metadata.
    """
    # Handle empty component (single point)
    if len(freeman.path) == 0 and len(freeman.children) == 0:
        return RootNode(
            NoneValue(),
            CoordValue(start),
            PaletteValue(frozenset(colors)),
        )

    program = encode_freeman_to_knode(freeman)
    return RootNode(
        program,
        CoordValue(start),
        PaletteValue(frozenset(colors)),
    )


# --- Starting Point Selection ---


def find_candidate_start_points(component: Coords) -> Coords:
    """
    Find candidate starting points for Freeman encoding.

    Selects structurally significant points that may yield
    more compressible encodings:
    - Bounding box corners (if in component)
    - Point closest to centroid

    Args:
        component: Set of coordinates forming the component.

    Returns:
        Set of candidate starting coordinates.
    """
    if not component:
        return set()

    starts: set[Coord] = set()
    col_coords, row_coords = zip(*component)

    # Bounding box
    col_min, col_max = min(col_coords), max(col_coords)
    row_min, row_max = min(row_coords), max(row_coords)

    # Add corners that are part of component
    corners = [
        Coord(col_min, row_min),
        Coord(col_max, row_min),
        Coord(col_min, row_max),
        Coord(col_max, row_max),
    ]
    starts.update(c for c in corners if c in component)

    # Add point closest to centroid
    col_centroid = sum(col_coords) // len(col_coords)
    row_centroid = sum(row_coords) // len(row_coords)

    def dist_to_centroid(coord: Coord) -> int:
        return (coord[0] - col_centroid) ** 2 + (coord[1] - row_centroid) ** 2

    closest = min(component, key=dist_to_centroid)
    starts.add(closest)

    return starts


# --- High-Level Encoding API ---


def syntax_tree_at(
    component: Coords,
    colors: Colors,
    start: Coord,
    traversal: TraversalModes = TraversalModes.DFS,
) -> RootNode[MoveValue]:
    """
    Encode a connected component as a syntax tree from a given start point.

    Args:
        component: Set of coordinates forming the component.
        colors: Color palette of the component.
        start: Starting coordinate for traversal.
        traversal: DFS or BFS traversal mode.

    Returns:
        RootNode encoding the component.

    Raises:
        ValueError: If start is not in component.
    """
    if start not in component:
        raise ValueError(f"Starting point {start} not in component")

    # Build Freeman tree via connectivity traversal
    def in_component(coord: Coord) -> bool:
        return coord in component

    freeman = encode_connected_component(start, in_component, traversal)

    return create_root_node(freeman, start, colors)


def encode_component_distribution(
    component: Coords,
    colors: Colors,
) -> tuple[RootNode[MoveValue], ...]:
    """
    Generate distribution of raw syntax trees for a component.

    Explores different starting points and traversal modes to find
    potentially more compressible encodings.

    Args:
        component: Set of coordinates forming the component.
        colors: Color palette of the component.

    Returns:
        Tuple of RootNodes sorted by bit_length (most compact first).
    """
    starts = find_candidate_start_points(component)

    # Explore all combinations of start points and traversal modes
    configurations = [(start, mode) for start in starts for mode in TraversalModes]

    trees = [
        syntax_tree_at(component, colors, start, mode) for start, mode in configurations
    ]

    return tuple(sorted(trees, key=lambda t: t.bit_length()))


def encode_component(
    component: Coords,
    colors: Colors,
) -> tuple[tuple[KNode[MoveValue], ...], tuple[KNode[MoveValue], ...]]:
    """
    Encode a component as a compressed distribution of syntax trees.

    Applies full compression pipeline:
    1. Generate raw distribution from different start points
    2. Extract nested patterns
    3. Co-symbolize across distribution

    Args:
        component: Set of coordinates forming the component.
        colors: Color palette of the component.

    Returns:
        Tuple of (compressed_distribution, symbol_table).
        Distribution is sorted by bit_length (most compact first).
    """
    symbol_table: list[KNode[MoveValue]] = []

    # Generate raw distribution
    raw = encode_component_distribution(component, colors)

    # Extract nested patterns from each tree
    nested = tuple(extract_nested_patterns(symbol_table, tree) for tree in raw)

    # Co-symbolize: find shared patterns across all trees
    symbolized, symbols = symbolize(nested, tuple(symbol_table))

    return (
        tuple(sorted(symbolized, key=lambda x: x.bit_length())),
        symbols,
    )


# --- Composite Structure ---


@dataclass
class UnionNode(Generic[T]):
    """
    Represents a component as union of single-color programs.

    Used for reconstruction after marginalization by color.
    The component is the union of shapes, with optional
    normalization (background) and shadowing information.

    Attributes:
        nodes: Set of KNodes representing component shapes.
        shadowed: Indices of points occluded by overlapping shapes.
        normalizing_node: Optional background/base shape.
    """

    nodes: set[KNode[T]]
    shadowed: set[int] | None = None
    normalizing_node: KNode[T] | None = None

    def bit_length(self) -> int:
        """Total description length of the union."""
        total = sum(node.bit_length() for node in self.nodes)
        if self.normalizing_node is not None:
            total += self.normalizing_node.bit_length()
        return total
