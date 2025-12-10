"""
Type definitions for ARC-AGI grid processing.

This module contains all types specific to the ARC (Abstraction and Reasoning
Corpus) domain: colors, coordinates, grids, and task data structures.

Coordinate Convention:
    All coordinates use (col, row) order, where:
    - col: x-axis, increases rightward (0 to width-1)
    - row: y-axis, increases downward (0 to height-1)

Color Convention:
    ARC uses integers 0-9 as colors, where 0 typically represents background.
"""

from collections.abc import Iterable, Set
from typing import NamedTuple, TypeVar, TypedDict, cast


# =============================================================================
# Color Types
# =============================================================================

type Color = int
"""Single ARC color (0-9)."""

type Colors = Set[Color]
"""Set of colors, typically representing a palette or object colors."""


# =============================================================================
# Grid Types
# =============================================================================

type ColorGrid = list[list[Color]]
"""2D grid indexed as grid[row][col] -> Color."""

type Mask = list[list[bool]]
"""Boolean mask indexed as mask[row][col] -> bool."""

type Grid = ColorGrid | Mask
"""Generic grid type."""


# =============================================================================
# Coordinate Types
# =============================================================================


class Coord(NamedTuple):
    """
    2D coordinate in (col, row) format.

    Note: This differs from grid indexing which uses [row][col].
    """

    col: int
    row: int


type Coords = Set[Coord]
"""Set of coordinates, typically representing a shape or region."""


class Point(NamedTuple):
    """Colored coordinate: a position with an associated color."""

    col: int
    row: int
    color: Color


type Points = Set[Point]
"""Set of colored points, representing a colored shape."""

type CoordsOrPoints = Points | Coords
"""Either colored or uncolored coordinate set."""


type Box = tuple[Coord, Coord]
"""Axis-aligned bounding box: (top_left, bottom_right) corners."""


class Proportions(NamedTuple):
    """Grid dimensions."""

    width: int
    height: int


# =============================================================================
# Type Utilities
# =============================================================================

_T = TypeVar("_T")


def ensure_all_instances(seq: Iterable[object], cls: type[_T]) -> Iterable[_T]:
    """Verify all items in sequence are instances of cls, return typed sequence."""
    if not all(isinstance(item, cls) for item in seq):
        raise TypeError(f"Not all items are instances of {cls}: {seq}")
    return cast(Iterable[_T], seq)


# =============================================================================
# Graph/Traversal Types
# =============================================================================

type Direction = str
"""Movement direction identifier (e.g., 'N', 'SE', 'right')."""

type Transition = tuple[Direction, Coord]
"""Graph edge: (direction_taken, destination_coordinate)."""


# =============================================================================
# Task Data Types
# =============================================================================

type Json = dict[str, None | int | str | bool | list["Json"] | dict[str, "Json"]]
"""Recursive JSON-compatible type."""


class Example(TypedDict):
    """Single input-output example pair."""

    input: ColorGrid
    output: ColorGrid


class TaskData(TypedDict):
    """Complete ARC task with train and test splits."""

    train: list[Example]
    test: list[Example]


# =============================================================================
# Dimension Aliases (for documentation clarity)
# =============================================================================

type Height = int
type Width = int
type Row = int
type Col = int


__all__ = [
    # Colors
    "Color",
    "Colors",
    # Grids
    "ColorGrid",
    "Mask",
    "Grid",
    # Coordinates
    "Coord",
    "Coords",
    "Point",
    "Points",
    "CoordsOrPoints",
    "Box",
    "Proportions",
    # Utilities
    "ensure_all_instances",
    # Graph
    "Direction",
    "Transition",
    # Task data
    "Json",
    "Example",
    "TaskData",
    # Dimension aliases
    "Height",
    "Width",
    "Row",
    "Col",
]
