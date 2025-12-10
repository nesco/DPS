"""
Type definitions for grid processing operations.

This module contains all custom types used throughout the grid processing library,
organized by their primary use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
    runtime_checkable,
)

# Basic type variables for generic operations
T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")

# Data Tyoe
type Json = dict[str, None | int | str | bool | list[Json] | dict[str, Json]]


class Example(TypedDict):
    input: ColorGrid
    output: ColorGrid


class TaskData(TypedDict):
    train: list[Example]
    test: list[Example]


# Color-related types
type Color = int
type Colors = Set[Color]

# Grid representations
type ColorGrid = list[list[Color]]  # Functional: grid[row][col] -> color
type Mask = list[list[bool]]  # Boolean mask: grid[row][col] -> is_selected
type Grid = ColorGrid | Mask  # Generic grid type


# Coordinate systems
class Coord(NamedTuple):
    col: int
    row: int


# type Coord = tuple[int, int]  # (col, row) coordinates
type Coords = Set[Coord]  # Set of coordinates
type Box = tuple[Coord, Coord]  # (top_left, bottom_right) corners


class Proportions(NamedTuple):
    width: int
    heigth: int


# Point system (extended coordinates)
# type Point = tuple[int, int, int]  # (col, row, color)
class Point(NamedTuple):
    col: int
    row: int
    color: int


type Points = Set[Point]  # Set of colored points

# Combined types
type CoordsGeneralized = Points | Coords  # Either colored or uncolored coordinates

# Graph traversal
type Trans = tuple[str, Coord]  # (direction, target_coordinate)

# Category theory
type Quotient[U, T] = Mapping[
    U, Set[T]
]  # Maps representatives to their equivalence classes

# Type aliases for improving code readability
Height = int
Width = int
Row = int
Col = int
Direction = str


@dataclass(frozen=True)
class GridObject:
    """Class containing all informations to represent a connected component"""

    colors: Colors
    coords: Coords


# Base interface for values with bit lengths
@dataclass(frozen=True)
class BitLengthAware(ABC):
    """Interface for values that can report their bit lengths."""

    @abstractmethod
    def bit_length(self) -> int:
        """Returns the bit length of this value."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns the string representation of this value."""
        pass


@dataclass(frozen=True)
class Primitive(BitLengthAware):
    """Base class for all primitives: bit length aware wrapping base values"""

    value: Any

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class KeyValue(Primitive):
    """
    Key for identifying positions in tree structures.

    Note: bit_length() returns 0. MDL costs are computed externally
    in the edit distance algorithms where context (sequence length,
    field count, etc.) is available.
    """

    value: (
        str | None | tuple[str | None, int]
    )  # int -> tuple, str -> dataclass, None -> Primitive or Frozen set

    def bit_length(self) -> int:
        return 0


TNode = TypeVar("TNode", bound="BitLengthAware")


@runtime_checkable
class Resolvable(Protocol[TNode]):
    """
    An element that can be resolved to a concrete BitLengthAware
    value given an external symbol table.
    """

    def resolve(self, symbol_table: Sequence[TNode]) -> TNode: ...

    def eq_ref(self, other: Resolvable) -> bool: ...

    """ Return True if both return to the same reference"""


def ensure_all_instances(seq: Iterable[object], cls: type[T]) -> Iterable[T]:
    if not all(isinstance(item, cls) for item in seq):
        raise TypeError(f"Not all items are instances of {cls}: {seq}")
    return cast(Iterable[T], seq)


__all__ = [
    # Type variables
    "T",
    "U",
    "P",
    # Basic types
    "Color",
    "Colors",
    # Grid types
    "ColorGrid",
    "Mask",
    "Grid",
    # Coordinate types
    "Coord",
    "Coords",
    "Box",
    "Proportions",
    # Point types
    "Point",
    "Points",
    # Combined types
    "CoordsGeneralized",
    # Graph types
    "Trans",
    # Category theory types
    "Quotient",
    # Type aliases
    "Height",
    "Width",
    "Row",
    "Col",
    "Direction",
    # BitLength
    "BitLengthAware",
    # Type guard
    "ensure_all_instances",
]
