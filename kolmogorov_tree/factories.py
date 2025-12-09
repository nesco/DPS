"""
Factory functions for creating common Kolmogorov Tree nodes.

This module provides convenience functions for creating nodes:
- create_move_node: Create a directional move primitive
- create_variable_node: Create a variable placeholder
- cv: Create a coordinate value
- create_moves_sequence: Create a sequence of moves from a direction string
- create_rect: Create a rectangle shape pattern
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
    """Creates a node for a directional move."""
    return PrimitiveNode(MoveValue(direction))


def create_variable_node(i: int) -> VariableNode:
    """Creates a VariableNode(VariableValue)."""
    return VariableNode(VariableValue(i))


def cv(r: int, c: int) -> CoordValue:
    """Creates a CoordValue."""
    return CoordValue(Coord(c, r))  # Coord(col, row)


def create_moves_sequence(directions: str) -> KNode | None:
    """Creates a sequence of moves from a direction string."""
    if not directions:
        return None
    moves = [create_move_node(int(d)) for d in directions]
    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len
                if (
                    all(
                        moves[i * pattern_len : (i + 1) * pattern_len] == pattern
                        for i in range(repeat_count)
                    )
                    and repeat_count > 1
                ):
                    return RepeatNode(
                        ProductNode(tuple(pattern)), CountValue(repeat_count)
                    )
    return ProductNode(tuple(moves))


def create_rect(height: int, width: int) -> KNode | None:
    """Creates a node for a rectangle shape."""
    if height < 2 or width < 2:
        return None
    first_row = "2" * (width - 1)  # Move right
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"  # Alternate left/right
        other_rows += "3" + direction * (width - 1)  # Down then horizontal
    return create_moves_sequence(first_row + other_rows)


__all__ = [
    "create_move_node",
    "create_variable_node",
    "cv",
    "create_moves_sequence",
    "create_rect",
]
