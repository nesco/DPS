"""
Grid can be totally or partially encoded in a lossless fashion into Asbract Syntax Trees.
The lossless part is essential here because ARC Corpus sets the problem into the low data regime.
First grids are marginalized into connected components of N colors. Those connected components are "shapes".

Shapes are then represented the following way: extracting a possible branching pathes with a DFS on the connectivity graph,
then representing thoses branching pathes as a non-deterministic "program", where the output are string representing moves à la  freeman chain codes

The lossless encoding used is basic pattern matching for (meta) repetitions through RepeatNode and SymbolNode.
The language formed by ASTs are simple enough an approximate version of kolmogorov complexity can be computed.
It helps choosing the most efficient encoding, which is the closest thing to a objective  proper representation
of the morphology.

The kolmogorov nodes are in a way the transitions of an automata over grid coordinates

"""

from dataclasses import dataclass
from typing import Generic, cast

# from ast import Moves
from freeman import (
    DIRECTIONS_FREEMAN,
    FreemanNode,
    King,
    TraversalModes,
    encode_connected_component,
)
from kolmogorov_tree import (
    CoordValue,
    CountValue,
    KNode,
    MoveValue,
    PaletteValue,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    T,
    VariableNode,
    expand_repeats,
    extract_nested_patterns,
    factorize_tuple,
    iterable_to_product,
    iterable_to_sum,
    postmap,
    premap,
    reverse_node,
    shift,
    symbolize,
)
from localtypes import Colors, Coord, Coords, Points
from utils.grid import coords_to_points, points_to_grid_colored


def moves_to_rect(moves: str) -> tuple[int, int] | None:
    """Detect if a move sequence forms a rectangle, returning (height, width) or None.
    Hypothesis: the rect is explored in a DFS fashion from the top-right corner
    """
    if not moves or len(moves) <= 2:
        return None
    width = moves.index("3") + 1 if "3" in moves else len(moves) + 1
    height = moves.count("3") + 1
    expected = "2" * (width - 1) + "".join(
        "3" + ("0" if i % 2 else "2") * (width - 1) for i in range(1, height)
    )
    return (
        (height, width)
        if moves == expected and height >= 2 and width >= 2
        else None
    )


def extract_moves_from_product(knode: KNode) -> str | None:
    """Extract move string from a ProductNode of PrimitiveNodes with MoveValues."""
    if isinstance(knode, ProductNode):
        node = cast(ProductNode, expand_repeats(knode))
        moves = []
        for child in node.children:
            if isinstance(child, PrimitiveNode) and isinstance(
                child.value, MoveValue
            ):
                moves.append(str(child.data))
            else:
                return None
        return "".join(moves)
    return None


# TO-DO: Make it work, or add a default symbol for rects
def detect_rect_node(node: KNode) -> KNode:
    """
    Replace a ProductNode with a RectNode if it forms a rectangle.
    Hypothesis: the rect is explored in a DFS fashion from the top-right corner
    """
    moves_str = str(expand_repeats(node))
    if moves_str:
        rect = moves_to_rect(moves_str)
        if rect:
            height, width = rect
            return RectNode(CountValue(height), CountValue(width))
    return node


def freeman_to_knode(freeman: FreemanNode) -> KNode[MoveValue]:
    """
    Encode the Freeman tree structure into a Kolmogorov Tree without compression.
    Hypothesis: A freeman tree has always either a non empty path or children
    """
    product = []

    # Step 1: Encode the main path
    moves = [PrimitiveNode(MoveValue(move)) for move in freeman.path]
    if moves:
        product.extend(moves)

    # Step 2: Recursively encode children
    children = iterable_to_sum(
        frozenset({freeman_to_knode(child) for child in freeman.children})
    )
    if children:
        product.append(children)

    product = iterable_to_product(product)

    if not product:
        raise ValueError("Freeman node had both empty path and children fields")

    return product


def encode_freeman_to_knode(freeman: FreemanNode) -> KNode[MoveValue]:
    """Encode a Freeman tree with compression (factorization and rectangle detection)."""
    uncompressed_knode = freeman_to_knode(freeman)

    factored_node = postmap(uncompressed_knode, factorize_tuple)
    # Apply rectangle detection
    factored_node = premap(uncompressed_knode, detect_rect_node)
    # Factorize the KTree
    # factored_node = postmap(factored_node, factorize_tuple)
    return factored_node


def freeman_node_to_root_node(
    freeman_node: FreemanNode, start_position: Coord, colors: Colors
) -> RootNode:
    """Encode a Freeman tree for a connected component into a RootNode."""
    program = encode_freeman_to_knode(freeman_node)
    return RootNode(
        program, CoordValue(start_position), PaletteValue(frozenset(colors))
    )


# Decoding
def decode_knode(
    knode: ProductNode[MoveValue]
    | SumNode[MoveValue]
    | PrimitiveNode[MoveValue]
    | RepeatNode[MoveValue]
    | RectNode,
    start: Coord,
) -> Coords:
    """Unfold a concrete KNode in the set of coords traversed by the pathes it represents."""
    coords = {start}

    def execute(knode: KNode, pos: Coord) -> Coords:
        """Supports SumNode not always at the end"""
        nonlocal coords
        match knode:
            case PrimitiveNode(MoveValue(direction)):
                delta = DIRECTIONS_FREEMAN[cast(King, direction)]
                new_pos = Coord(pos[0] + delta[0], pos[1] + delta[1])
                coords.add(new_pos)
                return {new_pos}
            case ProductNode(children):
                current = {pos}
                for child in children:
                    new_positions = set()
                    for p in current:
                        new_p = execute(child, p)
                        new_positions.update(new_p)
                    current = new_positions
                return current
            case SumNode(children) if len(children) == 1 and isinstance(
                next(iter(children)), RepeatNode
            ):
                repeat = next(iter(children))
                repeat = cast(RepeatNode, repeat)
                node, count = repeat.node, repeat.count
                if isinstance(count, VariableNode):
                    raise TypeError(
                        "Trying to uncompress an abstract Repeat node"
                    )
                N = count.value
                if N == 0:
                    raise ValueError(
                        "Repeat count should not be zero in iterator case"
                    )
                if N > 0:
                    shifts = range(N)
                else:
                    shifts = range(0, N, -1)
                new_positions = set()
                for k in shifts:
                    shifted_node = shift(node, k)
                    new_p = execute(shifted_node, pos)
                    new_positions.update(new_p)
                return new_positions
            case SumNode(children):
                new_positions = set()
                for child in children:
                    new_p = execute(child, pos)
                    new_positions.update(new_p)
                return new_positions
            case RepeatNode(node, count):
                if isinstance(count, VariableNode):
                    raise TypeError(
                        "Trying to uncompress an abstract Repeat node"
                    )
                count_value = count.value
                current = {pos}
                if count_value < 0:
                    # Handle negative count: alternate between node and its reverse
                    k = -count_value  # Number of iterations
                    for i in range(k):
                        # Even i: use original node; Odd i: use reversed node
                        node_to_execute = (
                            node if i % 2 == 0 else reverse_node(node)
                        )
                        new_positions = set()
                        for p in current:
                            new_p = execute(node_to_execute, p)
                            new_positions.update(new_p)
                        current = new_positions
                else:
                    # Positive count: repeat the node as before
                    for _ in range(count_value):
                        new_positions = set()
                        for p in current:
                            new_p = execute(node, p)
                            new_positions.update(new_p)
                        current = new_positions
                return current
            case RectNode(height, width):
                if isinstance(height, VariableNode) or isinstance(
                    width, VariableNode
                ):
                    raise TypeError(
                        "Trying to uncompress an abstract Repeat node"
                    )
                height = height.value
                width = width.value
                rect_points = [
                    Coord(pos[0] + j, pos[1] + i)
                    for i in range(height)
                    for j in range(width)
                ]
                coords.update(rect_points)
                return {pos}  # Final position remains the starting point
            case _:
                raise TypeError(f"Unsupported node type: {type(knode)}")

    final_positions = execute(knode, start)
    return coords


def decode_root(root: RootNode[MoveValue]) -> Points:
    node, position, colors = root.node, root.position, root.colors

    if isinstance(position, VariableNode) or isinstance(colors, VariableNode):
        raise TypeError("Root should not contain variables during decoding")

    if len(colors.value) != 1:
        raise ValueError("Root should be unicolored to be decoded")

    color = list(colors.value)[0]

    # Retrieve the coords
    match node:
        case (
            ProductNode()
            | SumNode()
            | PrimitiveNode()
            | RepeatNode()
            | RectNode()
        ):
            coords = decode_knode(node, position.value)
        case _:
            raise TypeError("Root's node should be a concrete KNode")

    return coords_to_points(coords, color)


@dataclass
class UnionNode(Generic[T]):
    """
    Represent a connected component by reconstructing it with the best set of single color programs.
    After marginalisation comes reconstruction, divide and conquer.
    """

    nodes: set[KNode[T]]
    shadowed: set[int] | None = None
    normalizing_node: KNode[T] | None = None

    def bit_length(self) -> int:
        # Maybe remove the cost of the shadowed nodes
        len_nodes = sum(node.bit_length() for node in self.nodes)
        return len_nodes + (
            0
            if self.normalizing_node is None
            else self.normalizing_node.bit_length()
        )


def get_potential_starting_points(component: Coords) -> Coords:
    """
    Return a list of special points which could serve as starting points to construct a Freeman Tree from a mask

    Args:
        object_coords: Coords

    Return:
        Coords: set of potential coordinates
    """

    starts = set()
    if not component:
        return starts

    # Step 1: Extract x and y coordinates
    col_coords, row_coords = zip(*component)

    # Step 2: Find the box boundaries:
    col_max, col_min = max(col_coords), min(col_coords)
    row_max, row_min = max(row_coords), min(row_coords)

    # Step 3:  Adding corner points if they are in the mask
    corners = [
        (col_min, row_min),
        (col_max, row_min),
        (col_min, row_max),
        (col_max, row_max),
    ]

    starts |= {corner for corner in corners if corner in component}

    # Step 4: The closest point to the approximate centroid
    col_centroid = sum(col_coords) // len(col_coords)
    row_centroid = sum(row_coords) // len(row_coords)

    def distance_to_centroid(coord):
        return (coord[0] - col_centroid) ** 2 + (coord[1] - row_centroid) ** 2

    closest_to_centroid = min(component, key=distance_to_centroid)
    starts.add(closest_to_centroid)

    return starts


def syntax_tree_at(
    component: Coords,
    colors: Colors,
    start: Coord,
    traversal_mode: TraversalModes = TraversalModes.DFS,
) -> RootNode[MoveValue]:
    """
    Transforms the mask of a connected component, with it's starting point, to a KNode of MoveValues
    """

    # Step 0: Validate the start point:
    if start not in component:
        raise ValueError(f"Starting point: {start}, noot in component")

    # Step 1: Build the Freeman tree of the component
    def is_coord_in_component(coord: Coord) -> bool:
        return coord in component

    freeman = encode_connected_component(
        start, is_coord_in_component, traversal_mode
    )

    # Step 2: Construct the RootNode
    root_node = freeman_node_to_root_node(freeman, start, colors)

    return root_node


def component_to_raw_syntax_tree_distribution(
    component: Coords, colors: Colors
) -> tuple[RootNode[MoveValue], ...]:
    """
    Args
        component: A connected component
        colors: A color palette

    Returns:

    """
    starts = get_potential_starting_points(component)

    configuration_space = {
        (start, traversal_mode)
        for start in starts
        for traversal_mode in TraversalModes
    }
    raw_syntax_tree_distribution = [
        syntax_tree_at(component, colors, start, traversal_mode)
        for start, traversal_mode in configuration_space
    ]

    return tuple(
        sorted(raw_syntax_tree_distribution, key=lambda st: st.bit_length())
    )

    # symbolized syntax tree distribution


def component_to_distribution(
    component: Coords, colors: Colors
) -> tuple[tuple[KNode[MoveValue], ...], tuple[KNode[MoveValue], ...]]:
    symbol_table = []

    # Encode a shape by its raw distribution
    raw_distribution = component_to_raw_syntax_tree_distribution(
        component, colors
    )

    # Compress nested patterns
    nested_distribution = tuple(
        extract_nested_patterns(symbol_table, syntax_tree)
        for syntax_tree in raw_distribution
    )

    # co-symbolize the entire distribution
    # Note: A greedy approach can be followed by symbolizing each syntax tree with it's own symbol table
    # Here, it will further compress represensation which have common templates in the distribution
    # The theoretical justification I give myself is that the true representation of the shape is the entire distribution,
    # not a singular syntax tree distribution
    symbolized_distribution, symbol_table = symbolize(
        tuple(nested_distribution), tuple(symbol_table)
    )

    # alternatively full_symbolization(raw_distribution) works also

    return symbolized_distribution, symbol_table


### tests


# Helper to convert grid to Freeman node (simplified)
def grid_to_freeman(grid, start, color) -> FreemanNode:
    """
    Convert a grid to a Freeman node representation using DFS traversal.

    Args:
        grid: A ColorGrid (list of lists of integers)
        start: A tuple (col, row) indicating the starting position
        color: A set of colors to consider (assumed singleton)

    Returns:
        FreemanNode: A node representing the grid's chain code
    """
    if len(colors) != 1:
        raise ValueError(
            "This implementation assumes a unicolored connected component"
        )
    color = next(iter(colors))  # Extract the single color from the set

    def is_valid(coord):
        col, row = coord
        # Check if the coordinate is within grid bounds
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            # Check if the grid value matches the color
            return grid[row][col] == color
        return False

    node = encode_connected_component(start, is_valid)
    return node


# Test function
def test_encode_decode(grid, start, colors, name):
    """
    Test the encode-decode round-trip for a grid.

    Args:
        grid: A ColorGrid to encode
        start: Starting position (col, row)
        colors: Set of colors to encode
        name: Name of the test case for printing
    """
    # Step 1: Encode
    freeman = grid_to_freeman(grid, start, colors)
    print(f"freeman :{freeman}")
    tree = freeman_node_to_root_node(freeman, start, colors)
    print(f"{name} Encoded Tree: {tree}")

    # Step 2: Decode
    points = decode_root(tree)
    decoded_grid = points_to_grid_colored(points)

    print(f"grid: {grid}")
    print(f"decoded grid: {decoded_grid}")
    # Step 3: Verify
    assert grid == decoded_grid, f"{name} round-trip failed"
    print(f"{name} passed round-trip test")


if __name__ == "__main__":
    # Test grids defined as lists of lists (ColorGrid format)
    grids = {
        "Simple Square": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "Two Squares": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        "Cross": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        "Rectangle": [[1, 1], [1, 1], [1, 1]],
    }

    # Run tests
    for name, grid in grids.items():
        start = (0, 0)  # Top-left corner
        if name == "Cross":
            start = (0, 1)
        colors = {1}  # Single color
        test_encode_decode(grid, start, colors, name)

    # Test ARC task

    # # Example ARC grid as a list of lists
    # task_grid = [
    #     [1, 0, 1],
    #     [0, 1, 0],
    #     [1, 0, 1]
    # ]
    # lattice = input_to_lattice(task_grid)  # Assume this accepts ColorGrid
    # ast = lattice.codes[0]  # First component's AST
    # points = decode_root(ast)
    # decoded_grid = points_to_grid_colored(points)
    # assert task_grid == decoded_grid, "ARC task round-trip failed"
    print("ARC task test passed")
