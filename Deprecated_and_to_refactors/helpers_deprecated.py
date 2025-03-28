"""
Repository of helper functions.

It sets basics operations on grids on two possible representations.
Either a functional one:
    - Grid: list[list[int]], [row][column] -> value
Or a set-centric one:
    - Points: set[Point], {(column, row, value)} / (column, row, value) -> {True, False}
"""

import json
import os
import pathlib
import time
import traceback
from collections import defaultdict
from functools import reduce, wraps
from itertools import combinations
from typing import (
    Callable,
    Concatenate,
    Optional,
    ParamSpec,
    overload,
)

from category import *
from grid import DIRECTIONS, GridOperations, grid_to_points
from localtypes import *

# Shortcuts
proportions = GridOperations.proportions
zeros = GridOperations.zeros

## Constants
# Define ANSI escape codes for the closest standard terminal colors
from constants import DATA, DEBUG


# set_to_quotient groups togetherm elements of the set into equivalence classes
# based on the projection operations
def set_to_quotient(
    projection: Callable[[T], U], object_set: set[T]
) -> Quotient:
    """
    Given a function, represent a set by a couple (quotient set, equivalence classes).
    The quotients are the dict keys and the equivalence classes the values.
    This construction needs an objective function that gives a representative to each element of
    the original set into the quotient set.
    @param object_set: set of objects
    @param projection: to each element of the original set, it assigns a representative in the quotient set
    """
    groups = defaultdict(set)
    for obj in object_set:
        groups[projection(obj)].add(obj)
    return dict(groups)


def quotient_to_set(quotient: Quotient) -> set[T]:
    return set.union(*quotient.values())


P = ParamSpec("P")

# Basic Monads


# DEBUG mode switch
def debug(fun: Callable) -> Callable:
    @wraps(fun)
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"Calling {fun.__name__} with args: {args}, kwargs: {kwargs}")
            result = fun(*args, **kwargs)
            print(f"{fun.__name__} returned: {result}")
            return result
        return fun(*args, **kwargs)

    return wrapper


# Maybe / Optional
def optional(
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[Optional[T], P], Optional[U]]:
    @wraps(func)
    def wrapper(arg: Optional[T], *args, **kwargs) -> Optional[U]:
        if arg is None:
            return None
        return func(arg, *args, **kwargs)

    return wrapper


@overload
def handle_elements(
    func: Callable[Concatenate[list[T], P], list[U]],
) -> Callable[Concatenate[list[T], P], list[U]]: ...
@overload
def handle_elements(
    func: Callable[Concatenate[list[T], P], list[U]],
) -> Callable[Concatenate[T, P], Optional[U]]: ...


def handle_elements(
    func: Callable[..., list[U]],
) -> Callable[..., list[U] | Optional[U]]:
    @wraps(func)
    def wrapper(arg: T | list[T], *args, **kwargs) -> list[U] | Optional[U]:
        if isinstance(arg, list):
            return func(arg, *args, **kwargs)
        else:
            result = func([arg], *args, **kwargs)
            return result[0] if len(result) == 1 else None

    return wrapper


@overload
def handle_lists(
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[T, P], U]: ...
@overload
def handle_lists(
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[list[T], P], list[U]]: ...


def handle_lists(func: Callable[..., U]) -> Callable[..., list[U] | U]:
    @wraps(func)
    def wrapper(arg: list[T] | T, *args, **kwargs) -> list[U] | U:
        if isinstance(arg, list):
            return [func(element, *args, **kwargs) for element in arg]
        return func(arg, args, **kwargs)

    return wrapper


## Basic decorator
def to_grid(func: Callable[[T], U]) -> Callable[[list[list[T]]], list[list[U]]]:
    @wraps(func)
    def wrapper(arg: list[list[T]]) -> list[list[U]]:
        height, width = len(arg), len(arg[0])
        return [
            [func(arg[row][col]) for col in range(width)]
            for row in range(height)
        ]

    return wrapper


def coordinate_shift(
    transformation_ls: list[tuple[Coord, Coord]],
) -> Optional[Coord]:
    "Given a list of coordinate changes, find if there is a unique shift"
    shifts = set()
    for (col1, row1), (col2, row2) in transformation_ls:
        shifts.add((col2 - col1, row2 - row1))
    if len(shifts) == 1:
        return shifts.pop()
    else:
        return None


def argmin_by_len(lst):
    return min(range(len(lst)), key=lambda i: len(lst[i]))


# Might be replaceable by a simple "in valid region"
def valid_coordinates(
    height: int,
    width: int,
    valid_region: Optional[set[Coord]],
    coordinates: Coord,
) -> bool:
    in_borders = 0 <= coordinates[0] < width and 0 <= coordinates[1] < height
    respecting_constraints = (
        coordinates in valid_region if valid_region else True
    )
    return in_borders & respecting_constraints


def available_transitions(
    is_valid: Callable[[Coord], bool], coordinates: Coord, moves: str
) -> list[Trans]:
    transitions = []
    col, row = coordinates
    for move in moves:
        nrow, ncol = row + DIRECTIONS[move][1], col + DIRECTIONS[move][0]
        ncoordinates = (ncol, nrow)
        if is_valid(ncoordinates):
            transitions.append((move, ncoordinates))
    return transitions


def list_to_grid(height, width, coordinates_ls):
    grid = zeros(height, width)
    for col, row in coordinates_ls:
        grid[row][col] = 1
    return grid


def grid_to_list(grid):
    width, height = proportions(grid)
    return [
        (j, i) for i in range(height) for j in range(width) if grid[i][j] == 1
    ]


### Function

# Basic functor

# Grid <-> Points functors and their own helper functions
# As Points is only construtced from Grid, it has no separate constructor

# region Box basics
# Box constructors
# def points_to_box(points: Points) -> Box:
#    rows, cols , _ = zip(*points)
#    row_min, row_max = min(rows), max(rows)
#    col_min, col_max = min(cols), max(cols)
#    return (row_min, col_min),  (row_max, col_max)


def touches_border(coords: CoordsGeneralized, box: Box) -> bool:
    (col_min, row_min), (col_max, row_max) = box
    return any(
        row == row_min or row == row_max or col == col_min or col == col_max
        for item in coords
        for col, row in [item[:2]]
    )


def is_coord_in_box(coord: Coord, box: Box) -> bool:
    col, row = coord
    (col_min, row_min), (col_max, row_max) = box

    return col_min <= col <= col_max and row_min <= row <= row_max


def is_box_included(box1: Box, box2: Box) -> bool:
    """Returns if box1 is include in box2"""
    corner_top_left, corner_bottom_right = box1
    return is_coord_in_box(corner_top_left, box2) and is_coord_in_box(
        corner_bottom_right, box2
    )


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
    for row in range(row_min, row_max + 1):
        grid_extract.append([])
        for col in range(col_min, col_max + 1):
            grid_extract[row - row_min].append(grid[row][col])

    return grid_extract


def create_centered_padded(grid, new_height, new_width):
    width, height = proportions(grid)
    grid_new = zeros(new_height, new_width)

    pad_height = max(0, new_height - height)
    pad_width = max(0, new_width - width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Fill in the original grid values
    for row in range(height):
        for col in range(width):
            grid_new[row + pad_top][col + pad_left] = grid[row][col]
    return grid_new


# Distance in grids represented as list[list[int]]
def distance_jaccard_grid(grid1: Grid, grid2: Grid):
    points1, points2 = grid_to_points(grid1), grid_to_points(grid2)
    return distance_jaccard(points1, points2)


def distance_jaccard_optimal_grid(grid1: Grid, grid2: Grid):
    return distance_jaccard_optimal(
        grid_to_points(grid1), grid_to_points(grid2)
    )


# Distances in grids represented as Points
def distance_jaccard(points1: Points, points2: Points) -> float:
    intersection = len(points1 & points2)
    union = len(points1 | points2)
    return 1 - intersection / union if union else 0


def distance_jaccard_optimal(
    points1: Points, points2: Points
) -> tuple[float, Coord]:
    min_distance = float("inf")
    min_shift = (0, 0)

    # Founding the bounding
    _, (col_max1, row_max1) = coords_to_box(points1)
    _, (col_max2, row_max2) = coords_to_box(points2)

    # Sliding grid2 over grid1
    for drow in range(-row_max2, row_max1 + 1):
        for dcol in range(-col_max2, col_max1 + 1):
            # Shift grid2
            shifted_points2 = {
                (col + dcol, row + drow, val) for col, row, val in points2
            }
            distance = distance_jaccard(points1, shifted_points2)
            if distance < min_distance:
                min_distance = distance
                min_shift = (drow, dcol)

    return min_distance, min_shift


def mask_colors(grid, mask):
    color_set = set()
    width_grid, height_grid = proportions(grid)
    width_mask, height_mask = proportions(mask)

    if height_grid != height_mask or width_grid != width_mask:
        raise Exception(
            "Grid and mask dimensions do not match: Grid: "
            + str((height_grid, width_grid))
            + "  Mask: "
            + str((height_mask, width_mask))
        )

    for row in range(height_mask):
        for col in range(width_mask):
            if mask[row][col] == 1:
                color_set.add(grid[row][col])
    return color_set


def grid_to_color_coords(grid: Grid) -> dict[Color, Coords]:
    width, height = proportions(grid)
    colors = set(
        [grid[row][col] for row in range(height) for col in range(width)]
    )
    colors_coords = {}
    for color in colors:
        color_coords = set()
        for row in range(height):
            for col in range(width):
                if grid[row][col] == color:
                    color_coords.add((col, row))
        colors_coords[color] = color_coords

    return colors_coords


def points_to_color_coords(points: Points) -> dict[Color, Coords]:
    _, _, colors = zip(*points)
    return {
        color: {(col, row) for col, row, value in points if value == color}
        for color in colors
    }


def ncolors_coords(colors_coords, min_n=2, max_n=10):
    colors = list(colors_coords.keys())
    return {
        frozenset(combo): set.union(*(colors_coords[c] for c in combo))
        for n in range(min_n, min(max_n, len(colors)) + 1)
        for combo in combinations(colors, n)
    }


def filter_by_color(grid, color):
    width, height = proportions(grid)
    grid_new = zeros(height, width)

    for i in range(height):
        for j in range(width):
            if grid[i][j] == color:
                grid_new[i][j] = 1

    return grid_new


def distance_levenshtein(chain1: str, chain2: str) -> int:
    m, n = len(chain1), len(chain2)

    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if chain1[i - 1] == chain2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # Return the Levenshtein distance
    return dp[m][n]


## Display


def clear_screen():
    # For Windows
    if os.name == "nt":
        _ = os.system("cls")
    # For macOS and Linux
    else:
        _ = os.system("clear")


def animate_grid(grids, delay=0.5):
    for grid in grids:
        clear_screen()
        GridOperations.print(grid)
        time.sleep(delay)


def print_binary_grid(grid):
    for row in grid:
        print(" ".join("1" if cell else "0" for cell in row))


def printf_binary_grid(grid):
    bold = "\033[1m"
    reset = "\033[0m"
    for row in grid:
        print(" ".join(f"{bold}1{reset}" if cell else "0" for cell in row))


## Data
"""
def read_path(path):
    with open(os.path.join(DATA, path), "r") as file:
        data = json.load(file)
    return data
"""


def get_all():
    path_dir = pathlib.Path(DATA + "/training")
    path_files = []
    uuids = []
    for json_file in path_dir.glob("*.json"):
        uuids.append(json_file.stem)
        path_files.append(json_file)
    data = []
    for path in path_files:
        with open(path, "r") as file:
            data.append(json.load(file))
    return data, uuids


def set_to_category1(sets: list[set], distance: Distance) -> Category:
    # Hypothesis: distance to None is intrinsic cost
    # Step 1: identify the invariants, and extract the variants
    invariants = reduce(set.intersection, sets)
    variants = [s - invariants for s in sets]

    # Step 2: Try to make associations through pairwise comparisons
    associations = defaultdict(lambda: defaultdict(set))
    for (i, set1), (j, set2) in combinations(enumerate(variants), 2):
        edit_matrix = [[(distance(a, b), a, b) for b in set2] for a in set1]

        while edit_matrix and any(edit_matrix):
            # min_dist, a, b = max(
            #    (item for row in edit_matrix for item in row if item),
            #        key=lambda x: distance(None, x[1]) + distance(None, x[2]) - x[0]
            #    )
            min_dist, a, b = min(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[0],
            )

            # If adding the association reduces the total cost, add it
            if min_dist < distance(None, a) + distance(None, b):
                associations[i][a].add((j, b))
                associations[j][b].add((i, a))

            # Remove processed items
            edit_matrix = [
                [item for item in row if item[1] != a and item[2] != b]
                for row in edit_matrix
                if any(item[1] != a for item in row)
            ]

    if DEBUG and False:
        for i, el in associations.items():
            print(f"Input {i}:")
            for a in el:
                print(f"Element associated to {a}:")
                for j, b in el[a]:
                    print(f"For input {j}: element {b}")
                    print(f"Distance: {distance(a, b)}")
    # Step 3: Remove associations that violates transitivity
    # You want to only retains cliques / complete subgraphes
    changed = True
    while changed:
        changed = False
        # For every sets, for each of it's elements
        # Go to the elements it's associated with and check
        # their are only associated with elements its associated with
        for i in associations:
            for a in list(associations[i].keys()):
                for j, b in list(associations[i][a]):
                    for k, c in list(associations[j][b]):
                        # (i, a) is not in itself as the identity transition is not saved
                        if (k, c) not in associations[i][a] and (k, c) != (
                            i,
                            a,
                        ):
                            # Transitivity violation found, remove the weakest link
                            links = [
                                (
                                    distance(None, a)
                                    + distance(None, b)
                                    - distance(a, b),
                                    (i, a),
                                    (j, b),
                                ),
                                (
                                    distance(None, b)
                                    + distance(None, c)
                                    - distance(b, c),
                                    (j, b),
                                    (k, c),
                                ),
                                (
                                    distance(None, a)
                                    + distance(None, c)
                                    - distance(a, c),
                                    (i, a),
                                    (k, c),
                                ),
                            ]
                            _, (x, y), (z, w) = min(links, key=lambda x: x[0])
                            associations[x][y].discard((z, w))
                            associations[z][w].discard((x, y))
                            changed = True

                            # If an association becomes empty, remove it
                            if not associations[x][y]:
                                del associations[x][y]
                            if not associations[z][w]:
                                del associations[z][w]

    if DEBUG and False:
        for i, el in associations.items():
            print(f"Input {i}:")
            for a in el:
                print(f"Element associated to {a}:")
                for j, b in el[a]:
                    print(f"For input {j}: element {b}")
                    print(f"Distance: {distance(a, b)}")
    # Step 4: Cluster the clique elements into equivalence classes
    cliques = []
    processed = set()
    for i in associations:
        for a in associations[i]:
            if (i, a) not in processed:
                clique = {(i, a)}
                to_process = [(i, a)]
                while to_process:
                    current = to_process.pop()
                    for associated in associations[current[0]][current[1]]:
                        if associated not in clique:
                            clique.add(associated)
                            to_process.append(associated)

                # Only keep clusters with one element from each set
                if len(clique) == len(sets) and len(
                    set(i for i, _ in clique)
                ) == len(sets):
                    cliques.append(clique)

                processed.update(clique)

    if DEBUG and False:
        for i, cluster in enumerate(cliques):
            print(f"\nClique n°{i}")
            for j, el in cluster:
                print(f"- From set n°{j} Element {el}")

    # Step 5: Create morphisms
    morphisms = {}

    for i, j in combinations(range(len(sets)), 2):
        morphisms[(i, j)] = morphisms[(j, i)] = {}

        # Identity on invariants
        for elem in invariants:
            morphisms[(i, j)][elem] = morphisms[(j, i)][elem] = elem

        # Clique-based mapping
        for clique in cliques:
            elem_i = next(a for k, a in clique if k == i)
            elem_j = next(a for k, a in clique if k == j)
            morphisms[(i, j)][elem_i] = elem_j
            morphisms[(j, i)][elem_j] = elem_i

        # Forgetful on the rest
        for elem in variants[i]:
            if elem not in morphisms[(i, j)]:
                morphisms[(i, j)][elem] = None

    category = Category(invariants, cliques, sets, morphisms)
    return category


def load(task="2dc579da.json"):
    data = read_path("training/" + task)
    inputs = [el["input"] for el in data["train"]]
    outputs = [el["output"] for el in data["train"]]
    return inputs, outputs


def load_final(task="2dc579da.json"):
    data = read_path("training/" + task)
    inputs = [el["input"] for el in data["train"]]
    outputs = [el["output"] for el in data["train"]]
    input_test = data["test"][0]["input"]
    output_test = data["test"][0]["output"]

    return inputs, outputs, input_test, output_test


#### Debug
def print_trace(e):
    print("".join(traceback.format_tb(e.__traceback__)))


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.6f} seconds to execute")
        return result

    return wrapper
