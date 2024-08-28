"""
Repository of helper functions.

It sets basics operations on grids on two possible representations.
Either a functional one:
    - Grid: List[List[int]], (row, column) -> value
Or a set-centric one:
    - Points: Set[Point], {(row, column, value)} / (row, column, value) -> {True, False}
"""
import os
import json
from re import X
import time
from pathlib import Path
from itertools import combinations


from dataclasses import dataclass, asdict
from typing import Any, List, Union, Optional, Iterator, Callable, Tuple, Set, Dict

## Constants
# Define ANSI escape codes for the closest standard terminal colors
COLORS = {
    0: "\033[40m",   # black
    1: "\033[44m",   # blue
    2: "\033[101m",  # red
    3: "\033[42m",   # green
    4: "\033[103m",  # yellow
    5: "\033[47m",   # white (for gray, best we can do)
    6: "\033[45m",   # magenta (for fuschia)
    7: "\033[43m",   # dark yellow (for orange, best we can do)
    8: "\033[46m",   # cyan (for teal)
    9: "\033[41m"    # dark red (for brown)
}

DIRECTIONS = {
    '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
    '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
}

MOVES = "01234567"

DATA = "../ARC-AGI/data"

## Types
# Proportions is equivalent data-wise to Coord but different meaning
# Beware of the off-by-one error
# A grid of bottom-right corner (max_row, max_col) will have proportions (height = max_row+1, width = max_col+1)

# Point is basically an unfolded (point: Point, value: int) into a trouple
GridColored = List[List[int]] # Functional representation of a grid: (row, col) -> val
Mask = List[List[bool]]
Grid = Union[GridColored, Mask]

Color = int

Coord = Tuple[int, int] # Coordinates
Coords = Set[Coord]
Box = Tuple[Coord, Coord]

Proportions = Tuple[int, int]

Point = Tuple[int, int, int] # Coordinates + color value
Points = Set[Point] # Set representation of a grid: {(row, col, val)} / (row, col, val) -> {True, False}

Trans = Tuple[str, Coord] # Transition
#


def coordinate_shift(transformation_ls: List[Tuple[Coord, Coord]]) -> Optional[Coord]:
    "Given a list of coordinate changes, find if there is a unique shift"
    shifts = set()
    for (row1, col1), (row2, col2) in transformation_ls:
        shifts.add((row2-row1, col2-col1))
    if len(shifts) == 1:
        return shifts.pop()
    else:
        return None

def argmin_by_len(lst):
    return min(range(len(lst)), key=lambda i: len(lst[i]))

# Might be replaceable by a simple "in valid region"
def valid_coordinates(height: int, width: int, valid_region: Optional[Set[Coord]], coordinates: Coord) -> bool:
    in_borders =  (0 <= coordinates[0] < height and 0 <= coordinates[1] < width)
    respecting_constraints = coordinates in valid_region if valid_region else True
    return in_borders & respecting_constraints
def available_transitions(is_valid: Callable[[Coord], bool], coordinates: Coord, moves: str) -> List[Trans]:
    transitions = []
    row, col = coordinates
    for move in moves:
        nrow, ncol = row + DIRECTIONS[move][0], col + DIRECTIONS[move][1]
        ncoordinates = (nrow, ncol)
        if is_valid(ncoordinates):
            transitions.append((move, ncoordinates))
    return transitions


class SpaceTopology():
    #directions = directions_chebyshev if chebyshev else directions_manhattan

    def __init__(self, height, width, constraints=None):
        self.height, self.width = height, width
        self.constraints = set(constraints) if constraints else None
        self.moves = "01234567"

        self.directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
        }

    def is_valid(self, coordinates):
        return valid_coordinates(self.height, self.width, self.constraints, coordinates)

    def transitions(self, coordinates, available=[]):
        neighbours = []
        row, col = coordinates
        for move in MOVES:
            nrow, ncol = row + DIRECTIONS[move][0], col + DIRECTIONS[move][1]
            ncoordinates = (nrow, ncol)
            if self.is_valid(ncoordinates):
                neighbours.append((move, ncoordinates))
        return neighbours

    def transitionsTower(self, coordinates, available=[]):
        neighbours = []
        row, col = coordinates
        for move in MOVES[:4]:
            nrow, ncol = row + DIRECTIONS[move][0], col + DIRECTIONS[move][1]
            ncoordinates = (nrow, ncol)
            if self.is_valid(ncoordinates):
                neighbours.append((move, ncoordinates))
        return neighbours

    def transitionsBishop(self, coordinates, available=[]):
        neighbours = []
        row, col = coordinates
        for move in MOVES[4:]:
            nrow, ncol = row + DIRECTIONS[move][0], col + DIRECTIONS[move][1]
            ncoordinates = (nrow, ncol)
            if self.is_valid(ncoordinates):
                neighbours.append((move, ncoordinates))
        return neighbours


class ColorTopology():

    def __init__(self):
        self.colours = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def is_valid(self, colour):
        return colour in self.colours

    def transitions(self, coordinate):
        neighbours = []
        for colour in self.colours:
            neighbours.append((str(colour), colour))
        return neighbours


def list_to_grid(height, width, coordinates_ls):
    grid = zeros(height, width)
    for row, col in coordinates_ls:
        grid[row][col] = 1
    return grid
def grid_to_list(grid):
    height, width = proportions(grid)
    return [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 1]

### Function


#region Grid Basics
# Elementary GridColored constructors
def zeros(height: int, width: int) -> GridColored:
    return [[0 for _ in range(width)] for _ in range(height)]
def ones(height: int, width: int) -> GridColored:
    return [[1 for _ in range(width)] for _ in range(height)]
def identity(height: int, width: int) -> GridColored:
    return [[1 if col == row else 0 for col in range(width)] for row in range(height)]
# Elementary grid functions
def proportions(grid: Grid) -> Proportions:
    height, width = len(grid), len(grid[0])
    return height, width
def copy(grid: Grid) -> Grid:
    height, width = proportions(grid)
    return [[grid[row][col] for col in range(width)] for row in range(height)]
# Elementary Mask constructors:
def falses(height: int, width:int) -> Mask:
    return [[False for _ in range(width)] for _ in range(height)]
def trues(height: int, width:int) -> Mask:
    return [[True for _ in range(width)] for _ in range(height)]
#endregion

#region Grid functors
def map_grid_colored(grid: GridColored, f: Callable[[int], int]) -> GridColored:
    height, width = proportions(grid)
    return [[f(grid[row][col]) for col in range(width)] for row in range(height)]

def map_mask(mask: Mask, f: Callable[[bool], bool]) -> Mask:
    height, width = proportions(mask)
    return [[f(mask[row][col]) for col in range(width)] for row in range(height)]
#endregion

# Grid <-> Points functors and their own helper functions
# As Points is only construtced from Grid, it has no separate constructor
def grid_to_points(grid: GridColored) -> Points:
    height, width = proportions(grid)
    return set([(row, col, grid[row][col]) for col in range(width) for row in range(height)])

def populate_grid_colored(grid: GridColored, points: Points) -> None:
    try:
        for row, col, val in points:
            grid[row][col] = val
    except IndexError as e:
        raise ValueError("The given list of points do not fit within the grid")
def proportions_points(points: Points) -> Proportions:
    rows, cols, _ = zip(*points)

    # Check for invalid values
    if any( row < 0 for row in rows) or any( col < 0 for col in cols):
        raise ValueError('Some points have negative rows or cols')

    # Get the proportions of the grid, beware of the off-by-one error
    height = max(rows) + 1
    width = max(cols) + 1
    return (height, width)

def proportions_coords(coords: Coords) -> Proportions:
    rows, cols = zip(*coords)

    # Check for invalid values
    if any( row < 0 for row in rows) or any( col < 0 for col in cols):
        raise ValueError('Some points have negative rows or cols')

    # Get the proportions of the grid, beware of the off-by-one error
    height = max(rows) + 1
    width = max(cols) + 1
    return (height, width)

def points_to_grid_colored(points: Points, height: Optional[int] = None, width: Optional[int] = None) -> GridColored:
    """
    Fit the points either in a given grid size, or in the smallest grid possible.
    """
    rows, cols, vals = zip(*points)

    # Check for invalid values
    if any( row < 0 for row in rows) or any( col < 0 for col in cols):
        raise ValueError('Some points have negative rows or cols')

    if not all( 0 <= val < 9 for val in vals ):
        raise ValueError('Some points have a color outside the [0, 9] range')

    # Get the proportions of the grid, beware of the off-by-one error
    nheight = max(rows) + 1
    nwidth = max(cols) + 1

    match (height, width):
        case None, None:
            height, width = nheight, nwidth
        case None, _:
            if width < nwidth:
                raise ValueError(f"Given width: {width} is too small, it should be at least: {nwidth}")
            height = nheight
        case _, None:
            if height < nheight:
                raise ValueError(f"Given height: {height} is too small, it should be at least: {nheight}")
            width = nwidth
        case _, _:
            if width < nwidth:
                raise ValueError(f"Given width: {width} is too small, it should be at least: {nwidth}")
            if height < nheight:
                raise ValueError(f"Given height: {height} is too small, it should be at least: {nheight}")

    # Constructing the scaffold, and filling it with the extracted points
    grid = zeros(height, width)
    populate_grid_colored(grid, points)

    return grid

#region Box basics
# Box constructors
def points_to_box(points: Points) -> Box:
    rows, cols , _ = zip(*points)
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    return (row_min, col_min),  (row_max, col_max)

def grid_to_box(grid: Grid) -> Box:
    return proportions_to_box(proportions(grid))
def proportions_to_box(prop: Proportions, corner_top_right: Coord = (0, 0)):
    row_min, col_min = corner_top_right
    height, width = prop
    return (row_min, col_min), (row_min + height-1, col_min + width-1)
def box_to_proportions(box: Box):
    (row_min, col_min), (row_max, col_max) = box
    return (row_max - row_min + 1, col_max - col_min + 1)



# indice work
def get_indices(height, width, bit, pos):
    ls_indice = []
    for i in range(height):
        if (i//2**pos) % 2 == bit:
            ls_indice.append(i)
    return ls_indice

def select(grid, bit, pos):
    size = len(grid)
    indice_ls = get_indices(size, size, bit, pos)
    indices = set()
    for i in indice_ls:
        for j in range(len(grid[0])):
            indices.add((i,j))
    return indices


def set_val(grid, indice_ls, val=1):
    for i, j in indice_ls:
        grid[i][j] = 1
# Simple perators on grids

def extract(grid, box):
    if box == None:
        return None

    top_left, bottom_right = box
    row_min, row_max = top_left[0], bottom_right[0]
    col_min, col_max = top_left[1], bottom_right[1]

    grid_extract = []
    for row in range(row_min, row_max+1):
        grid_extract.append([])
        for col in range(col_min, col_max+1):
            #grid_extract[row - row_min] = grid[row].copy()
            grid_extract[row - row_min].append(grid[row][col])

    return grid_extract
def create_centered_padded(grid, new_height, new_width):
    height, width = proportions(grid)
    grid_new = zeros(new_height, new_width)

    pad_height = max(0, new_height - height)
    pad_width = max(0, new_width - width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    #Fill in the original grid values
    for row in range(height):
        for col in range(width):
            grid_new[row + pad_top][col + pad_left] = grid[row][col]
    return grid_new

# Distance in grids represented as List[List[int]]
def distance_jaccard_grid(grid1: Grid, grid2: Grid):
    points1, points2 = grid_to_points(grid1), grid_to_points(grid2)
    return distance_jaccard(points1, points2)

def distance_jaccard_optimal_grid(grid1: Grid, grid2: Grid):
    return distance_jaccard_optimal(grid_to_points(grid1), grid_to_points(grid2))
# Distances in grids represented as Points
def distance_jaccard(points1: Points, points2: Points) -> float:
    intersection = len(points1 & points2)
    union = len(points1 | points2)
    return 1 - intersection / union if union else 0

def distance_jaccard_optimal(points1: Points, points2: Points) -> float:
    min_distance = float('inf')

    # Founding the bounding
    _, (row_max1, col_max1) = points_to_box(points1)
    _, (row_max2, col_max2) = points_to_box(points2)

    # Sliding grid2 over grid1
    for drow in range(-row_max2, row_max1+1):
        for dcol in range(-col_max2, col_max1+1):
            # Shift grid2
            shifted_points2 = {(row+drow, col+dcol, val) for row, col, val in points2}
            distance = distance_jaccard(points1, shifted_points2)
            min_distance = min(min_distance, distance)

    return min_distance

def mask_colors(grid, mask):
    color_set = set()
    height_grid, width_grid = proportions(grid)
    height_mask, width_mask = proportions(mask)

    if height_grid != height_mask or width_grid != width_mask:
        raise Exception("Grid and mask dimensions do not match: Grid: " +
            str((height_grid, width_grid)) + "  Mask: " + str((height_mask, width_mask)))

    for row in range(height_mask):
        for col in range(width_mask):
            if mask[row][col] == 1:
                color_set.add(grid[row][col])
    return color_set

def mask_to_coords(mask: Mask) -> Coords:
    height, width = proportions(mask)
    return set([(row, col) for row in range(height) for col in range(width) if mask[row][col]])

def populate_mask(mask: Mask, coords: Coords) -> None:
    try:
        for row, col in coords:
            mask[row][col] = True
    except IndexError as e:
        raise ValueError("The given list of points do not fit within the grid")
def coords_to_mask(coords: Coords, height: Optional[int] = None, width: Optional[int] = None) -> Mask:
    nheight, nwidth = proportions_coords(coords)

    match (height, width):
        case None, None:
            height, width = nheight, nwidth
        case None, _:
            if width < nwidth:
                raise ValueError(f"Given width: {width} is too small, it should be at least: {nwidth}")
            height = nheight
        case _, None:
            if height < nheight:
                raise ValueError(f"Given height: {height} is too small, it should be at least: {nheight}")
            width = nwidth
        case _, _:
            if width < nwidth:
                raise ValueError(f"Given width: {width} is too small, it should be at least: {nwidth}")
            if height < nheight:
                raise ValueError(f"Given height: {height} is too small, it should be at least: {nheight}")

    # Constructing the scaffold, and filling it with the extracted points
    mask = falses(height, width)
    populate_mask(mask, coords)

    return mask

def mask_to_grid(mask: Mask, color_map: Tuple[Color, Color] = (1, 0)) -> Grid:
    """
    Returns the first color if Truen the second if false
    """
    height, width = proportions(mask)
    return [[color_map[0] if mask[row][col] else color_map[1] for col in range(width)] for row in range(height)]

def grid_to_color_coords(grid: Grid) -> Dict[Color, Coords]:
    height, width = proportions(grid)
    colors = set([grid[row][col] for row in range(height) for col in range(width)])
    colors_coords = {}
    for color in colors:
        color_coords = set()
        for row in range(height):
            for col in range(width):
                if grid[row][col] == color:
                    color_coords.add((row, col))
        colors_coords[color] = color_coords

    return colors_coords
def points_to_color_coords(points: Points) -> Dict[Color, Coords]:
    _, _, colors = zip(*points)
    return {color: {(row, col) for row, col, value in points if value == color} for color in colors}

def ncolors_coords(colors_coords, min_n=2, max_n=10):
    colors = list(colors_coords.keys())
    return {
            frozenset(combo): set.union(*(colors_coords[c] for c in combo))
            for n in range(min_n, min(max_n, len(colors)) + 1)
            for combo in combinations(colors, n)
        }

def coords_to_points(coords: Coords, color: Color = 1):
    return set([(row, col, color) for row, col in coords])

def filter_by_color(grid, color):
    height, width = proportions(grid)
    grid_new = zeros(height, width)

    for i in range(height):
        for j in range(width):
            if grid[i][j] == color:
                grid_new[i][j] = 1

    return grid_new



def split_by_color_deprecated(grid):
    """Create for each color a mask, i.e a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids

## Display
def print_colored_grid(grid):
    height = len(grid)
    width = len(grid[0])

    print(f"{COLORS[5]}   "*(width+2), "\033[0m")

    for row in range(height):

        print(f"{COLORS[5]}   ", end="")
        for col in range(width):
            # Print with the selected color
            print(f"{COLORS[grid[row][col]]}   ", end="")
        # Reset color and move to next line
        print(f"{COLORS[5]}   ", "\033[0m")

    print(f"{COLORS[5]}   "*(width+2), "\033[0m")
def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')
def animate_grid(grids, delay=0.5):
    for grid in grids:
            clear_screen()
            print_colored_grid(grid)
            time.sleep(delay)
def print_binary_grid(grid):
    for row in grid:
        print(' '.join('1' if cell else '0' for cell in row))

def printf_binary_grid(grid):
    bold = '\033[1m'
    reset = '\033[0m'
    for row in grid:
        print(' '.join(f'{bold}1{reset}' if cell else '0' for cell in row))
## Data
def read_path(path):
    with open(os.path.join(DATA, path), 'r') as file:
        data = json.load(file)
    return data

def get_all():
    path_dir = Path(DATA + '/training')
    path_files = []
    uuids = []
    for json_file in path_dir.glob('*.json'):
        uuids.append(json_file.stem)
        path_files.append(json_file)
    data = []
    for path in path_files:
        with open(path, 'r') as file:
            data.append(json.load(file))
    return data, uuids

def load(task = "2dc579da.json"):
    data = read_path('training/' + task)
    inputs = [el['input'] for el in data['train']]
    outputs = [el['output'] for el in data['train']]
    return inputs, outputs
