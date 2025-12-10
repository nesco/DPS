"""
Type definitions for grid hierarchy analysis.

This module contains types for representing the hierarchical structure
of ARC grids after decomposition into connected components.
"""

from dataclasses import dataclass

from arc.types import Colors, Coords


@dataclass(frozen=True, slots=True)
class GridObject:
    """
    A connected component in a grid.

    Represents a maximal set of adjacent pixels that share color membership
    in a particular color set. The component is identified by its colors
    and the coordinates it occupies.

    Attributes:
        colors: The minimal color set that produces this component.
        coords: The coordinates occupied by this component.
    """

    colors: Colors
    coords: Coords


__all__ = ["GridObject"]
