"""
Repository of helper functions.
"""
import os
import json
from re import X
import time
from pathlib import Path

from dataclasses import dataclass, asdict
from typing import Any, List, Union, Optional, Iterator, Callable, Tuple, Set

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

# Types
Coord = Tuple[int, int] # Coordinates
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

def bfs(coordinates, topology):
    queue = [('', coordinates)]
    seen = set([coordinates])
    traversal = []

    while queue:
        path, element = queue.pop(0)
        traversal.append((path, element))
        # First Tower then Bishop
        for move, ncoordinates in topology.transitionsTower(element):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                queue.append((path + move, ncoordinates))
        for move, ncoordinates in topology.transitionsBishop(element):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                queue.append((path + move, ncoordinates))

    return traversal, seen

def dfs(coordinates, topology):
    stack = [('', coordinates)]
    seen = set([coordinates])  # Mark the starting point as seen immediately
    traversal = []
    while stack:
        path, element = stack.pop()  # Use pop() instead of pop(0) for DFS
        traversal.append((path, element))
        # First Tower then Bishop
        for move, ncoordinates in reversed(topology.transitionsBishop(element)):  # Reverse to maintain original order
            if ncoordinates not in seen:
                seen.add(ncoordinates)  # Mark as seen as soon as we discover it
                stack.append((path + move, ncoordinates))
        for move, ncoordinates in reversed(topology.transitionsTower(element)):  # Reverse to maintain original order
            if ncoordinates not in seen:
                seen.add(ncoordinates)  # Mark as seen as soon as we discover it
                stack.append((path + move, ncoordinates))

    return traversal, seen

def list_to_grid(height, width, coordinates_ls):
    grid = zeros(height, width)
    for row, col in coordinates_ls:
        grid[row][col] = 1
    return grid
def grid_to_list(grid):
    height, width = proportions(grid)
    return [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 1]

def construct_grid(start, node, topology):
    grid = zeros(topology.height, topology.width)
    populate(grid, start, node)
    return grid

def ast_to_grid(start, node, topology, show_construction = False):
    """
    Note: The show construction follows a DFS order, making a bfs path looks sloppy
    """
    grid = zeros(topology.height, topology.width)
    construction = [] if show_construction else None
    populate(grid, start, node, construction)
    if construction:
        animate_grid(construction)
    return grid
## Function
# Research
def extract_coordinates(grid):
    height, width = proportions(grid)
    return [((i, j), grid[i][j]) for i in range(height) for j in range(width)]

def marginalize(coordinates_ls):
    dimensions = len(coordinates_ls[0])
    margin_spaces = []
    for dim in range(dimensions):
        space_dim = {}
        for i, coordinates in enumerate(coordinates_ls):
            ncoordinates = coordinates[:dim] + coordinates[dim+1:]
            if len(ncoordinates) == 1:
                ncoordinates = ncoordinates[0]
            if coordinates[dim] in space_dim:
                space_dim[coordinates[dim]] += [ncoordinates]
            else:
                space_dim[coordinates[dim]] = [ncoordinates]
        margin_spaces.append(space_dim)
    return margin_spaces

    #seen =


# Simple grid creations
def zeros(height, width):
    return [[0 for _ in range(width)] for _ in range(height)]

def ones(height, width):
    return [[1 for _ in range(width)] for _ in range(height)]
def copy(grid):
    height, width = proportions(grid)
    return [[grid[row][col] for col in range(width)] for row in range(height)]
def identity(height, width):
    return [[1 if col == row else 0 for col in range(width)] for row in range(height)]

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
def proportions(grid):
    height, width = len(grid), len(grid[0])
    return height, width
def prop_box(box):
    (row1, col1), (row2, col2) = box
    return (row2 - row1, col2 - col1)

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

def colors_extract(grid):
    height, width = proportions(grid)
    colors_unique = set()
    for row in range(height):
        for col in range(width):
            colors_unique.add(grid[row][col])

    return colors_unique


def filter_by_color(grid, color):
    height, width = proportions(grid)
    grid_new = zeros(height, width)

    for i in range(height):
        for j in range(width):
            if grid[i][j] == color:
                grid_new[i][j] = 1

    return grid_new



def split_by_color(grid):
    """Create for each color a mask, i.e a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids

## Display
def print_dict(dictionary):
    def get_string(val, shift_amount=0):
        string = ""
        shift = '\t' * shift_amount

        if isinstance(val, list):
            string += "\n"
            for el in val:
                if isinstance(el, dict):
                    string += shift + get_string(el, shift_amount+1) + "\n"
                else:
                    string += shift + str(el) + "\n"
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, dict):
                    string_row = shift + "- "  + str(key) + "\n"
                    string_row += get_string(value, shift_amount+1)
                    string += string_row
                else:
                    string += shift + key + ":" + get_string(value) + "\n"
        return string
    print(get_string(dictionary))

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

## Serialization
def extract_serialized_elements(chain_code):
    start, code = chain_code.split(':') if ':' in chain_code else (chain_code, "")
    start_row, start_col = None, None if start == "" else map(int, start.split(","))
    return (start_row, start_col), code

def serialize(mask_connected, chebyshev=True):
    """
    Serialize a connected component into a Freeman Chain Code representation.

    Args:
        mask_connected (List[List[int]]): 2D binary mask representing the connected component.
        chebyshev (bool): If True, use 8-connectivity; otherwise, use 4-connectivity.

        Returns:
        str: Freeman Chain Code representation of the connected component.

    Thus the encoding use to represent the directions are:
    4-connectivity:
          0
        1 • 3
          2

    8-connectivity:
        1 0 7
        2 • 6
        3 4 5
    """
    height, width = proportions(mask_connected)
    visited = zeros(height, width)
    moves = "0123" if not chebyshev else "01234567"
    #directions = directions_chebyshev if chebyshev else directions_manhattan
    directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
    }

    def find_start():
        return next(
            ((row, col) for row in range(height) for col in range(width)
             if mask_connected[row][col] == 1),
            None
        )


    def is_valid(row, col):
           return (0 <= row < height and 0 <= col < width and
                   mask_connected[row][col] == 1 and visited[row][col] == 0)

    def get_inverse_move(move):
        return str((int(move) + (4 if chebyshev else 2)) % len(moves))

    def dfs(row, col):
        visited[row][col] = 1
        paths = []
        for move in moves:
            nrow, ncol = row + directions[move][0], col + directions[move][1]
            if is_valid(nrow, ncol):
                path, backtrack = dfs(nrow, ncol)
                paths.append(([move] + path, backtrack + [get_inverse_move(move)]))

        if not paths:
            return [], []

        paths.sort(key=lambda x: len(x[1]), reverse=True)
        main_path, main_backtrack = paths[0]
        other_paths = [path + backtrack for path, backtrack in paths[1:]]

        return sum(other_paths, []) + main_path, main_backtrack

    start = find_start()
    if not start:
        return ""

    encoding, _ = dfs(*start)
    return f"{start[0]},{start[1]}:{''.join(encoding)}"

def unserialize(mask, chain_code, chebyshev=True):
    (start_row, start_col), code = extract_serialized_elements(chain_code)

    directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
    }
    curr_row, curr_col = start_row, start_col
    mask[curr_row][curr_col] = 1
    for move in code:
        drow, dcol = directions[move]
        curr_row += drow
        curr_col += dcol
        mask[curr_row][curr_col] = 1
def code_compression(code):
    if not code:
        return '', ''

    morphology = code[0]
    coordinates = ''
    count = 1

    for i in range(1, len(code)):
        if code[i] == code[i-1]:
            count += 1
        else:
            coordinates += str(count)
            morphology += code[i]
            count = 1

    coordinates += str(count)
    return morphology, coordinates

def uncompress(morphology, coordinates):
    code = ''
    for i, direction in enumerate(morphology):
        code += direction * int(coordinates[i])

    return code
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
