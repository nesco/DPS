"""
Type definitions for grid processing operations.

This module contains all custom types used throughout the grid processing library,
organized by their primary use cases.
"""

from __future__ import annotations
from typing import TypeVar, ParamSpec

# Basic type variables for generic operations
T = TypeVar('T')
U = TypeVar('U')
P = ParamSpec('P')

# Color-related types
Color = int
Colors = set[Color]

# Grid representations
GridColored = list[list[Color]]  # Functional: grid[row][col] -> color
Mask = list[list[bool]]         # Boolean mask: grid[row][col] -> is_selected
Grid = GridColored | Mask # Generic grid type

# Coordinate systems
Coord = tuple[int, int]         # (col, row) coordinates
Coords = set[Coord]             # Set of coordinates
Box = tuple[Coord, Coord]       # (top_left, bottom_right) corners
Proportions = tuple[int, int]   # (width, height) of a grid

# Point system (extended coordinates)
Point = tuple[int, int, int]    # (col, row, color)
Points = set[Point]             # Set of colored points

# Combined types
CoordsGeneralized = Points | Coords  # Either colored or uncolored coordinates

# Graph traversal
Trans = tuple[str, Coord]       # (direction, target_coordinate)

# Category theory
Quotient = dict[U, set[T]]      # Maps representatives to their equivalence classes

# Type aliases for improving code readability
Height = int
Width = int
Row = int
Col = int
Direction = str

__all__ = [
    # Type variables
    'T', 'U', 'P',

    # Basic types
    'Color', 'Colors',

    # Grid types
    'GridColored', 'Mask', 'Grid',

    # Coordinate types
    'Coord', 'Coords', 'Box', 'Proportions',

    # Point types
    'Point', 'Points',

    # Combined types
    'CoordsGeneralized',

    # Graph types
    'Trans',

    # Category theory types
    'Quotient',

    # Type aliases
    'Height', 'Width', 'Row', 'Col', 'Direction'
]
