"""
Factory functions for creating Kolmogorov Tree nodes.

Functions:
    create_move_node(direction) - Create a directional move primitive
    create_variable_node(i)     - Create a variable placeholder
    cv(r, c)                    - Create a coordinate value
    create_moves_sequence(s)    - Create moves from direction string
    create_rect(h, w)           - Create a rectangle pattern
"""

from __future__ import annotations

from localtypes import Coord

from kolmogorov_tree.nodes import (
    KNode,
    PrimitiveNode,
    ProductNode,
    RepeatNode,
    VariableNode,
)
from kolmogorov_tree.primitives import (
    CoordValue,
    CountValue,
    MoveValue,
    VariableValue,
)


def create_move_node(direction: int) -> PrimitiveNode:
    """Creates a PrimitiveNode with a MoveValue (0-7 for 8 directions)."""
    return PrimitiveNode(MoveValue(direction))


def create_variable_node(i: int) -> VariableNode:
    """Creates a VariableNode with index i."""
    return VariableNode(VariableValue(i))


def cv(r: int, c: int) -> CoordValue:
    """Creates a CoordValue from row and column."""
    return CoordValue(Coord(c, r))


def create_moves_sequence(directions: str) -> KNode | None:
    """
    Creates a node from direction digits (0-7).

    Detects repeating patterns and compresses them into RepeatNodes.
    Returns None for empty string.
    """
    if not directions:
        return None
    moves = [create_move_node(int(d)) for d in directions]

    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len
                all_match = all(
                    moves[i * pattern_len : (i + 1) * pattern_len] == pattern
                    for i in range(repeat_count)
                )
                if all_match and repeat_count > 1:
                    return RepeatNode(
                        ProductNode(tuple(pattern)), CountValue(repeat_count)
                    )
    return ProductNode(tuple(moves))


def create_rect(height: int, width: int) -> KNode | None:
    """
    Creates a rectangle drawing pattern using moves.

    Returns None if height or width < 2. Uses boustrophedon (snake) pattern.
    """
    if height < 2 or width < 2:
        return None
    first_row = "2" * (width - 1)
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"
        other_rows += "3" + direction * (width - 1)
    return create_moves_sequence(first_row + other_rows)


__all__ = [
    "create_move_node",
    "create_variable_node",
    "cv",
    "create_moves_sequence",
    "create_rect",
]
