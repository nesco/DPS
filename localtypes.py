"""
Type definitions for grid processing operations.

This module contains all custom types used throughout the grid processing library,
organized by their primary use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

# Basic type variables for generic operations
T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")

# Color-related types
Color = int
Colors = set[Color]

# Grid representations
ColorGrid = list[list[Color]]  # Functional: grid[row][col] -> color
Mask = list[list[bool]]  # Boolean mask: grid[row][col] -> is_selected
Grid = ColorGrid | Mask  # Generic grid type

# Coordinate systems
Coord = tuple[int, int]  # (col, row) coordinates
Coords = set[Coord]  # Set of coordinates
Box = tuple[Coord, Coord]  # (top_left, bottom_right) corners
Proportions = tuple[int, int]  # (width, height) of a grid

# Point system (extended coordinates)
Point = tuple[int, int, int]  # (col, row, color)
Points = set[Point]  # Set of colored points

# Combined types
CoordsGeneralized = Points | Coords  # Either colored or uncolored coordinates

# Graph traversal
Trans = tuple[str, Coord]  # (direction, target_coordinate)

# Category theory
Quotient = dict[U, set[T]]  # Maps representatives to their equivalence classes

# Type aliases for improving code readability
Height = int
Width = int
Row = int
Col = int
Direction = str


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
]
