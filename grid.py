r"""
Grid Processing Library

A comprehensive library for handling different grid representations and transformations.
Supports functional (Grid) and set-centric (Points) representations of grids.

Operations occurs on four main data format: Grid, Mask, Points, and Coords
    1\ GridOperations
    2\ MaskOperations
    3\ PointsOperations
    4\ CoordsOperations
"""

from localtypes import Grid, ColorGrid, Proportions, Mask, Points, Coords, Color, CoordsGeneralized, Box, Coord
from constants import COLORS
from typing import Callable, Optional

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

# Functions (basic utils)

def unpack_coords(coords: CoordsGeneralized) -> tuple[list[int], list[int]]:
    cols, rows, *_ = zip(*coords)
    return (cols, rows)

def proportions_to_box(prop: Proportions, corner_top_right: Coord = (0, 0)):
    col_min, row_min = corner_top_right
    width, height = prop
    return (col_min, row_min), (col_min + width - 1, row_min + height - 1)

def box_to_proportions(box: Box) -> Proportions:
    (col_min, row_min), (col_max, row_max) = box
    return (row_max - row_min + 1, col_max - col_min + 1)

# Grid Base Operations
class GridOperations:
    """Basic grid operations and constructors"""

    # Constructors
    @staticmethod
    def zeros(height: int, width: int) -> ColorGrid:
        return [[0 for _ in range(width)] for _ in range(height)]

    @staticmethod
    def ones(height: int, width: int) -> ColorGrid:
        return [[1 for _ in range(width)] for _ in range(height)]

    @staticmethod
    def identity(height: int, width: int) -> ColorGrid:
        return [[1 if col == row else 0 for col in range(width)]
                for row in range(height)]

    @staticmethod
    def copy(grid: Grid) -> Grid:
        width, height = GridOperations.proportions(grid)
        return [[grid[row][col] for col in range(width)]
                for row in range(height)]

    @staticmethod
    def from_mask(mask: Mask, color_map: tuple[Color, Color] = (1, 0)) -> Grid:
        """
        Returns the first color if True the second if false
        """
        width, height = proportions(mask)
        return [
            [color_map[0] if mask[row][col] else color_map[1] for col in range(width)]
            for row in range(height)
        ]

    @staticmethod
    def from_points(
        points: Points, height: Optional[int] = None, width: Optional[int] = None
    ) -> ColorGrid:
        """
        Fit the points either in a given grid size, or in the smallest grid possible.
        """
        cols, rows, vals = zip(*points)

        # Check for invalid values
        if any(row < 0 for row in rows) or any(col < 0 for col in cols):
            raise ValueError("Some points have negative rows or cols")

        if not all(0 <= val < 9 for val in vals):
            raise ValueError("Some points have a color outside the [0, 9] range")

        # Get the proportions of the grid, beware of the off-by-one error
        nheight = max(rows) + 1
        nwidth = max(cols) + 1

        match (width, height):
            case None, None:
                width, height = nwidth, nheight
            case None, _:
                if height < nheight:
                    raise ValueError(
                        f"Given height: {height} is too small, it should be at least: {nheight}"
                    )
                height = nheight
            case _, None:
                if width < nwidth:
                    raise ValueError(
                        f"Given width: {width} is too small, it should be at least: {nwidth}"
                    )
                width = nwidth
            case _, _:
                if width < nwidth:
                    raise ValueError(
                        f"Given width: {width} is too small, it should be at least: {nwidth}"
                    )
                if height < nheight:
                    raise ValueError(
                        f"Given height: {height} is too small, it should be at least: {nheight}"
                    )

        # Constructing the scaffold, and filling it with the extracted points
        grid = GridOperations.zeros(height, width)
        GridOperations.populate(grid, points)

        return grid

    # Operations
    @staticmethod
    def proportions(grid: Grid) -> Proportions:
        height, width = len(grid), len(grid[0])
        return width, height

    @staticmethod
    def box(grid: Grid) -> Box:
        return proportions_to_box(GridOperations.proportions(grid))

    @staticmethod
    def map(grid: ColorGrid, f: Callable[[int], int]) -> ColorGrid:
        width, height = proportions(grid)
        return [[f(grid[row][col]) for col in range(width)] for row in range(height)]

    @staticmethod
    def populate(grid: ColorGrid, points: Points) -> None:
        try:
            for col, row, val in points:
                grid[row][col] = val
        except IndexError as e:
            raise ValueError("The given list of points do not fit within the grid")

    @staticmethod
    def print(grid: ColorGrid):
        height = len(grid)
        width = len(grid[0])

        print(f"{COLORS[5]}   " * (width + 2), "\033[0m")

        for row in range(height):
            print(f"{COLORS[5]}   ", end="")
            for col in range(width):
                # Print with the selected color
                print(f"{COLORS[grid[row][col]]}   ", end="")
            # Reset color and move to next line
            print(f"{COLORS[5]}   ", "\033[0m")

        print(f"{COLORS[5]}   " * (width + 2), "\033[0m")

class MaskOperations:
    """Basic mask operations and constructors"""

    # Constructors
    @staticmethod
    def falses(height: int, width: int) -> Mask:
        return [[False for _ in range(width)] for _ in range(height)]

    @staticmethod
    def trues(height: int, width: int) -> Mask:
        return [[True for _ in range(width)] for _ in range(height)]

    @staticmethod
    def from_coords(
        coords: Coords, height: Optional[int] = None, width: Optional[int] = None
    ) -> Mask:
        nwidth, nheight = coords_to_proportions(coords)

        match (width, height):
            case None, None:
                width, height = nwidth, nheight
            case None, _:
                if height < nheight:
                    raise ValueError(
                        f"Given height: {height} is too small, it should be at least: {nheight}"
                    )
                height = nheight
            case _, None:
                if width < nwidth:
                    raise ValueError(
                        f"Given width: {width} is too small, it should be at least: {nwidth}"
                    )
                width = nwidth
            case _, _:
                if width < nwidth:
                    raise ValueError(
                        f"Given width: {width} is too small, it should be at least: {nwidth}"
                    )
                if height < nheight:
                    raise ValueError(
                        f"Given height: {height} is too small, it should be at least: {nheight}"
                    )

        # Constructing the scaffold, and filling it with the extracted points
        mask = falses(height, width)
        MaskOperations.populate(mask, coords)

        return mask

    # Operations
    @staticmethod
    def populate(mask: Mask, coords: Coords) -> None:
        try:
            for col, row in coords:
                mask[row][col] = True
        except IndexError as e:
            raise ValueError("The given list of points do not fit within the grid")

    @staticmethod
    def map_mask(mask: Mask, f: Callable[[bool], bool]) -> Mask:
        width, height = proportions(mask)
        return [[f(mask[row][col]) for col in range(width)] for row in range(height)]

class PointsOperations:
    """Operations for point-based grid representation"""

    # Constructors
    @staticmethod
    def from_grid(grid: ColorGrid) -> Points:
        width, height = proportions(grid)
        return set(
            [(col, row, grid[row][col]) for col in range(width) for row in range(height)]
        )

    @staticmethod
    def from_coords(coords: Coords, color: Color = 1):
        return set([(col, row, color) for col, row in coords])

    # Operations
    @staticmethod
    def proportions(points: Points) -> Proportions:
        cols, rows, _ = zip(*points)

        # Check for invalid values
        if any(row < 0 for row in rows) or any(col < 0 for col in cols):
            raise ValueError("Some points have negative rows or cols")

        # Get the proportions of the grid, beware of the off-by-one error
        height = max(rows) + 1
        width = max(cols) + 1
        return (width, height)

class CoordsOperations:
    """Basic coords operations and constructors"""

    # Constructors
    @staticmethod
    def from_mask(mask: Mask) -> Coords:
        width, height = proportions(mask)
        return set(
            [(col, row) for row in range(height) for col in range(width) if mask[row][col]]
        )

    @staticmethod
    def from_points(points: Points) -> Coords:
        return set((col, row) for col, row, color in points)

    # Operations
    @staticmethod
    def proportions(coords: Coords) -> Proportions:
        cols, rows = unpack_coords(coords)

        # Check for invalid values
        if any(row < 0 for row in rows) or any(col < 0 for col in cols):
            raise ValueError("Some points have negative rows or cols")

        # Get the proportions of the grid, beware of the off-by-one error
        height = max(rows) + 1
        width = max(cols) + 1
        return (width, height)

    @staticmethod
    def box(coords: CoordsGeneralized) -> Box:
        cols, rows = unpack_coords(coords)
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        return (col_min, row_min), (col_max, row_max)
# Constructors as Functors

## grid <> points
grid_to_points = PointsOperations.from_grid
points_to_grid_colored = GridOperations.from_points

## coords <> mask
coords_to_mask = MaskOperations.from_coords
mask_to_coords = CoordsOperations.from_mask

## mask <> grid
mask_to_grid = GridOperations.from_mask

## points <> coords
points_to_coords = CoordsOperations.from_points
coords_to_points = PointsOperations.from_coords
