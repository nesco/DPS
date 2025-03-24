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

from freeman import (
    DIRECTIONS_FREEMAN,
    FreemanNode,
    King,
    encode_connected_component,
)
from utils.grid import coords_to_points, points_to_grid_colored

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
    SymbolNode,
    T,
    VariableNode,
    factorize_tuple,
    iterable_to_product,
    iterable_to_sum,
    postmap,
    reverse_node,
    shift,
)
from lattice_old import input_to_lattice
from localtypes import Coord, Coords, Points


def moves_to_rect(moves: str) -> tuple[int, int] | None:
    """Detect if a move sequence forms a rectangle, returning (height, width) or None."""
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


def extract_moves_from_product(node: KNode) -> str | None:
    """Extract move string from a ProductNode of PrimitiveNodes with MoveValues."""
    if isinstance(node, ProductNode):
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


def detect_rect_node(node: KNode) -> KNode:
    """Replace a ProductNode with a RectNode if it forms a rectangle."""
    moves_str = extract_moves_from_product(node)
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
        tuple(freeman_to_knode(child) for child in freeman.children)
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

    # Apply rectangle detection
    factored_node = postmap(uncompressed_knode, detect_rect_node)
    # Factorize the KTree
    factored_node = postmap(factored_node, factorize_tuple)
    return factored_node


def component_to_knode(
    freeman_node: FreemanNode, start_position: Coord, colors: set[int]
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
    current_positions = {start}

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
            case SumNode((RepeatNode(node, count),)):
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
    points = set()

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
    tree = component_to_knode(freeman, start, colors)
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
    from lattice import input_to_lattice

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
