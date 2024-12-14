"""
Grid Processing Library

A comprehensive library for handling different grid representations and transformations.
Supports functional (Grid) and set-centric (Points) representations of grids.
"""

from dataclasses import dataclass
from collections import defaultdict
from enum import StrEnum
from functools import wraps, reduce
from itertools import combinations
import json
import os
import pathlib
import time
import traceback

from localtypes import Grid, GridColored, Proportions

# Constants

DIRECTIONS = {
    "0": (-1, 0),  # left
    "1": (0, -1),  # up
    "2": (1, 0),   # right
    "3": (0, 1),   # down
    "4": (-1, -1), # up-left
    "5": (1, -1),  # up-right
    "6": (1, 1),   # down-right
    "7": (-1, 1),  # down-left
}

MOVES = "01234567"

# Grid Base Operations
class GridOperations:
    """Basic grid operations and constructors"""

    @staticmethod
    def zeros(height: int, width: int) -> GridColored:
        return [[0 for _ in range(width)] for _ in range(height)]

    @staticmethod
    def ones(height: int, width: int) -> GridColored:
        return [[1 for _ in range(width)] for _ in range(height)]

    @staticmethod
    def identity(height: int, width: int) -> GridColored:
        return [[1 if col == row else 0 for col in range(width)]
                for row in range(height)]

    @staticmethod
    def proportions(grid: Grid) -> Proportions:
        height, width = len(grid), len(grid[0])
        return width, height

    @staticmethod
    def copy(grid: Grid) -> Grid:
        width, height = GridOperations.proportions(grid)
        return [[grid[row][col] for col in range(width)]
                for row in range(height)]

class PointsOperations:
    """Operations for point-based grid representation"""
    pass
