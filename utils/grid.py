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

from re import M
from collections.abc import Callable, Mapping

from constants import COLORS
from localtypes import (
    Box,
    Color,
    ColorGrid,
    Coord,
    Coords,
    CoordsGeneralized,
    Grid,
    Mask,
    Point,
    Points,
    Proportions,
)

# Constants

DIRECTIONS = {
    "0": (-1, 0),  # left
    "1": (0, -1),  # up
    "2": (1, 0),  # right
    "3": (0, 1),  # down
    "4": (-1, -1),  # up-left
    "5": (1, -1),  # up-right
    "6": (1, 1),  # down-right
    "7": (-1, 1),  # down-left
}

MOVES = "01234567"


# helpers
def matrix_to_proportions(matrix: list[list]) -> Proportions:
    height, width = len(matrix), len(matrix[0])
    return Proportions(width, height)


# Functions (basic utils)


def unpack_coords(coords: CoordsGeneralized) -> tuple[list[int], list[int]]:
    cols, rows, *_ = zip(*coords)
    return (list(cols), list(rows))


def proportions_to_box(prop: Proportions, corner_top_right: Coord = Coord(0, 0)) -> Box:
    col_min, row_min = corner_top_right
    width, height = prop
    return Coord(col_min, row_min), Coord(col_min + width - 1, row_min + height - 1)


def box_to_proportions(box: Box) -> Proportions:
    (col_min, row_min), (col_max, row_max) = box
    return Proportions(row_max - row_min + 1, col_max - col_min + 1)


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
        return [
            [1 if col == row else 0 for col in range(width)]
            for row in range(height)
        ]

    @staticmethod
    def copy(grid: Grid) -> Grid:
        width, height = GridOperations.proportions(grid)
        return [
            [grid[row][col] for col in range(width)] for row in range(height)
        ]

    @staticmethod
    def from_mask(mask: Mask, color_map: tuple[Color, Color] = (1, 0)) -> Grid:
        """
        Returns the first color if True the second if false
        """
        width, height = MaskOperations.proportions(mask)
        return [
            [
                color_map[0] if mask[row][col] else color_map[1]
                for col in range(width)
            ]
            for row in range(height)
        ]

    @staticmethod
    def from_points(
        points: Points, height: int | None = None, width: int | None = None
    ) -> ColorGrid:
        """
        Fit the points either in a given grid size, or in the smallest grid possible.
        """
        cols, rows, vals = zip(*points)

        # Check for invalid values
        if any(row < 0 for row in rows) or any(col < 0 for col in cols):
            raise ValueError("Some points have negative rows or cols")

        if not all(0 <= val < 9 for val in vals):
            raise ValueError(
                "Some points have a color outside the [0, 9] range"
            )

        # Get the proportions of the grid, beware of the off-by-one error
        nheight = max(rows) + 1
        nwidth = max(cols) + 1

        # Set default values if None
        final_width = width if width is not None else nwidth
        final_height = height if height is not None else nheight

        if final_width < nwidth:
            raise ValueError(
                f"Given width: {final_width} is too small, it should be at least: {nwidth}"
            )
        if final_height < nheight:
            raise ValueError(
                f"Given height: {final_height} is too small, it should be at least: {nheight}"
            )
        # Constructing the scaffold, and filling it with the extracted points
        grid = GridOperations.zeros(final_height, final_width)
        GridOperations.populate(grid, points)

        return grid

    # Operations
    @staticmethod
    def proportions(grid: Grid) -> Proportions:
        return matrix_to_proportions(grid)

    @staticmethod
    def box(grid: Grid) -> Box:
        return proportions_to_box(GridOperations.proportions(grid))

    @staticmethod
    def map(grid: ColorGrid, f: Callable[[int], int]) -> ColorGrid:
        width, height = GridOperations.proportions(grid)
        return [
            [f(grid[row][col]) for col in range(width)] for row in range(height)
        ]

    @staticmethod
    def populate(grid: ColorGrid, points: Points) -> None:
        try:
            for col, row, val in points:
                grid[row][col] = val
        except IndexError:
            raise ValueError(
                "The given list of points do not fit within the grid"
            )

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
        coords: Coords, height: int | None = None, width: int | None = None
    ) -> Mask:
        nwidth, nheight = CoordsOperations.proportions(coords)

        width = nwidth if width is None else width
        height = nheight if height is None else height

        if width < nwidth:
            raise ValueError(
                f"Given width: {width} is too small, it should be at least: {nwidth}"
            )
        if height < nheight:
            raise ValueError(
                f"Given height: {height} is too small, it should be at least: {nheight}"
            )

        # Constructing the scaffold, and filling it with the extracted points
        mask = MaskOperations.falses(height, width)
        MaskOperations.populate(mask, coords)

        return mask

    @staticmethod
    def proportions(mask: Mask) -> Proportions:
        return matrix_to_proportions(mask)

    # Operations
    @staticmethod
    def populate(mask: Mask, coords: Coords) -> None:
        try:
            for col, row in coords:
                mask[row][col] = True
        except IndexError:
            raise ValueError(
                "The given list of points do not fit within the grid"
            )

    @staticmethod
    def map_mask(mask: Mask, f: Callable[[bool], bool]) -> Mask:
        width, height = MaskOperations.proportions(mask)
        return [
            [f(mask[row][col]) for col in range(width)] for row in range(height)
        ]


class PointsOperations:
    """Operations for point-based grid representation"""

    # Constructors
    @staticmethod
    def from_grid(grid: ColorGrid) -> Points:
        width, height = GridOperations.proportions(grid)
        return set(
            [
                Point(col, row, grid[row][col])
                for col in range(width)
                for row in range(height)
            ]
        )

    @staticmethod
    def from_coords(coords: Coords, color: Color = 1):
        return set([Point(col, row, color) for col, row in coords])

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
        return Proportions(width, height)


class CoordsOperations:
    """Basic coords operations and constructors"""

    # Constructors

    @staticmethod
    def from_grid(grid: ColorGrid) -> dict[Color, Coords]:
        width, height = GridOperations.proportions(grid)
        colors = set(
            grid[row][col] for row in range(height) for col in range(width)
        )
        color_dict = {}
        for color in colors:
            color_coords = set(
                (col, row)
                for row in range(height)
                for col in range(width)
                if grid[row][col] == color
            )
            color_dict[color] = color_coords
        return color_dict

    @staticmethod
    def from_mask(mask: Mask) -> Coords:
        width, height = MaskOperations.proportions(mask)
        return set(
            [
                Coord(col, row)
                for row in range(height)
                for col in range(width)
                if mask[row][col]
            ]
        )

    @staticmethod
    def from_points_erase(points: Points) -> Coords:
        return set(Coord(col, row) for col, row, color in points)

    @staticmethod
    def from_points(points: Points) -> Mapping[Color, Coords]:
        colors = set(color for col, row, color in points)
        coords_dict = {
            color_key: set(
                Coord(col, row) for col, row, color in points if color == color_key
            )
            for color_key in colors
        }
        return coords_dict

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
        return Proportions(width, height)

    @staticmethod
    def box(coords: CoordsGeneralized) -> Box:
        cols, rows = unpack_coords(coords)
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        return Coord(col_min, row_min), Coord(col_max, row_max)


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
points_to_coords = CoordsOperations.from_points_erase
coords_to_points = PointsOperations.from_coords

## points > coords_dict
points_to_coords_by_color = CoordsOperations.from_points
grid_to_coords_by_color = CoordsOperations.from_grid


## Grid <> Proportions
grid_to_proportions = GridOperations.proportions
coords_to_proportions = CoordsOperations.proportions
