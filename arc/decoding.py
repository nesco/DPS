"""
ARC syntax tree decoding to grid coordinates.

Transforms Kolmogorov syntax trees back into pixel coordinates,
executing the encoded move sequences to reconstruct shapes.

Decoding pipeline:
    RootNode → KNode → Coords → Points

Key functions:
    - decode_root: Decode a single RootNode to Points
    - decode_knode: Decode composite structures (sums/products of roots)
    - execute_moves: Execute KNode to produce coordinates
"""

from collections import defaultdict
from typing import cast

from freeman import DIRECTIONS_FREEMAN, King
from kolmogorov_tree import (
    KNode,
    MoveValue,
    NoneValue,
    PaletteValue,
    CoordValue,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    VariableNode,
    reverse_node,
    shift,
)
from localtypes import (
    Coord,
    Coords,
    Point,
    Points,
)
from utils.grid import coords_to_points


# Type alias for decodable move nodes
type DecodableNode = (
    ProductNode[MoveValue]
    | SumNode[MoveValue]
    | PrimitiveNode[MoveValue]
    | RepeatNode[MoveValue]
    | RectNode
    | NoneValue
)


def execute_moves(knode: DecodableNode, start: Coord) -> Coords:
    """
    Execute a KNode to produce the set of traversed coordinates.

    Interprets the KNode as a program that generates coordinates
    by following move directions from the starting point.

    Args:
        knode: KNode representing move sequence.
        start: Starting coordinate.

    Returns:
        Set of all coordinates visited during execution.

    Raises:
        TypeError: If knode contains unresolved VariableNodes.
    """
    visited: set[Coord] = {start}

    def execute(node: KNode | NoneValue, pos: Coord) -> set[Coord]:
        """Execute node from position, return final positions."""
        nonlocal visited

        match node:
            case NoneValue():
                return {pos}

            case PrimitiveNode(MoveValue(direction)):
                delta = DIRECTIONS_FREEMAN[cast(King, direction)]
                new_pos = Coord(pos[0] + delta[0], pos[1] + delta[1])
                visited.add(new_pos)
                return {new_pos}

            case ProductNode(children):
                # Sequential execution: each child continues from previous positions
                current = {pos}
                for child in children:
                    next_positions: set[Coord] = set()
                    for p in current:
                        next_positions.update(execute(child, p))
                    current = next_positions
                return current

            case SumNode(children) if _is_iterator_sum(children):
                # Special case: SumNode containing single RepeatNode = iterator
                return _execute_iterator(children, pos, execute, visited)

            case SumNode(children):
                # Branching: execute all children from same position
                final: set[Coord] = set()
                for child in children:
                    final.update(execute(child, pos))
                return final

            case RepeatNode(inner, count):
                if isinstance(count, VariableNode):
                    raise TypeError(
                        "Cannot execute abstract RepeatNode with VariableNode count"
                    )
                return _execute_repeat(inner, count.value, pos, execute, visited)

            case RectNode(height, width):
                if isinstance(height, VariableNode) or isinstance(width, VariableNode):
                    raise TypeError(
                        "Cannot execute abstract RectNode with VariableNode dimensions"
                    )
                return _execute_rect(height.value, width.value, pos, visited)

            case _:
                raise TypeError(f"Unsupported node type for execution: {type(node)}")

    execute(knode, start)
    return visited


def _is_iterator_sum(children: frozenset) -> bool:
    """Check if SumNode contains a single RepeatNode (iterator pattern)."""
    return len(children) == 1 and isinstance(next(iter(children)), RepeatNode)


def _execute_iterator(
    children: frozenset,
    pos: Coord,
    execute,
    visited: set[Coord],
) -> set[Coord]:
    """Execute iterator pattern: SumNode with single RepeatNode."""
    repeat = cast(RepeatNode, next(iter(children)))
    inner, count = repeat.node, repeat.count

    if isinstance(count, VariableNode):
        raise TypeError("Cannot execute abstract iterator with VariableNode count")

    n = count.value
    if n == 0:
        raise ValueError("Iterator repeat count should not be zero")

    shifts = range(n) if n > 0 else range(0, n, -1)
    final: set[Coord] = set()

    for k in shifts:
        shifted = shift(inner, k)
        final.update(execute(shifted, pos))

    return final


def _execute_repeat(
    inner: KNode,
    count: int,
    pos: Coord,
    execute,
    visited: set[Coord],
) -> set[Coord]:
    """Execute repeat: iterate inner node count times."""
    current = {pos}

    if count < 0:
        # Negative count: alternate between node and reversed node
        for i in range(-count):
            node_to_run = inner if i % 2 == 0 else reverse_node(inner)
            next_positions: set[Coord] = set()
            for p in current:
                next_positions.update(execute(node_to_run, p))
            current = next_positions
    else:
        # Positive count: repeat node sequentially
        for _ in range(count):
            next_positions = set()
            for p in current:
                next_positions.update(execute(inner, p))
            current = next_positions

    return current


def _execute_rect(
    height: int,
    width: int,
    pos: Coord,
    visited: set[Coord],
) -> set[Coord]:
    """Execute RectNode: fill rectangular area."""
    col, row = pos
    rect_coords = [
        Coord(col + dc, row + dr) for dr in range(height) for dc in range(width)
    ]
    visited.update(rect_coords)
    return {pos}  # RectNode doesn't change final position


def decode_root(root: RootNode[MoveValue]) -> Points:
    """
    Decode a single RootNode to colored points.

    Args:
        root: RootNode with concrete position and single color.

    Returns:
        Set of colored points representing the decoded shape.

    Raises:
        TypeError: If root contains unresolved variables.
        ValueError: If root has multiple colors (not unicolored).
    """
    node, position, colors = root.node, root.position, root.colors

    if isinstance(position, VariableNode) or isinstance(colors, VariableNode):
        raise TypeError("Cannot decode root with unresolved variables")

    if len(colors.value) != 1:
        raise ValueError(
            f"Root must be unicolored for decoding, got {len(colors.value)} colors"
        )

    color = next(iter(colors.value))

    # Validate node type
    if not isinstance(
        node,
        NoneValue | ProductNode | SumNode | PrimitiveNode | RepeatNode | RectNode,
    ):
        raise TypeError(f"Invalid node type in root: {type(node)}")

    coords = execute_moves(node, position.value)
    return coords_to_points(coords, color)


def decode_knode(
    knode: ProductNode[MoveValue] | SumNode[MoveValue] | RootNode[MoveValue],
) -> Points:
    """
    Decode composite KNode structures to points.

    Handles top-level structures:
    - RootNode: Single unicolored shape
    - SumNode: Union of shapes (branching)
    - ProductNode: Layered shapes (later colors occlude earlier)

    Args:
        knode: Top-level KNode to decode.

    Returns:
        Set of colored points.

    Raises:
        ValueError: If knode contains patterns (unresolved variables).
    """

    def execute(node: KNode[MoveValue]) -> Points:
        match node:
            case RootNode(inner, position, colors) if isinstance(
                position, VariableNode
            ) or isinstance(colors, VariableNode):
                raise ValueError(f"Cannot decode pattern RootNode: {node}")

            case RootNode(inner, position, colors) if (
                isinstance(colors, PaletteValue) and len(colors.value) != 1
            ):
                raise ValueError(f"RootNode must be unicolored: {colors}")

            case RootNode(inner, position, colors):
                assert isinstance(colors, PaletteValue)
                assert isinstance(position, CoordValue)
                if not isinstance(
                    inner,
                    NoneValue
                    | ProductNode
                    | SumNode
                    | PrimitiveNode
                    | RepeatNode
                    | RectNode,
                ):
                    raise ValueError(f"Invalid node in RootNode: {inner}")
                coords = execute_moves(inner, position.value)
                color = next(iter(colors.value))
                return coords_to_points(coords, color)

            case SumNode(children):
                # Union of all children
                all_points: set[Point] = set()
                for child in children:
                    all_points.update(execute(child))
                return frozenset(all_points)

            case ProductNode(children):
                # Layered: later children occlude earlier at same position
                points_by_coord: defaultdict[Coord, list[Point]] = defaultdict(list)

                for child in children:
                    for point in execute(child):
                        col, row, _ = point
                        points_by_coord[Coord(col, row)].append(point)

                # Keep only the last (topmost) point at each coordinate
                return frozenset(points[-1] for points in points_by_coord.values())

            case _:
                raise ValueError(f"Invalid top-level node: {type(node)}")

    return execute(knode)


# Backward-compatible alias
decode_knode_coords = execute_moves
