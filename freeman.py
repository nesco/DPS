
from typing import Any, List, Union, Optional, Iterator, Callable, Set, Tuple, Dict, NewType, Literal, Final
from typing import cast

from collections import deque

from helpers import Coord, Coords
from helpers import MOVES
from lattice import *
from dataclasses import dataclass

# Directions

Tower = Literal[0, 1, 2, 3]
Bishop = Literal[4, 5, 6, 7]
King = Union[Tower, Bishop]

NUM_DIRECTIONS: Final[int] = 8
NUM_DIRECTIONS_ORTHOGONAL: Final[int] = 4

TOWER: Final[List[Tower]] = [0, 1, 2, 3]
BISHOP: Final[List[Bishop]] = [4, 5, 6, 7]
KING: Final[List[King]] = [0, 1, 2, 3, 4, 5, 6, 7]

DIRECTIONS: Final[Dict[King, Coord]] = {
    0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1),
    4: (-1, -1), 5: (1, -1), 6: (1, 1), 7: (-1, 1)
}

DIRECTIONS_ARROW: Final[Dict[King, str]] = {
    0: "←", 1: "↑", 2: "→", 3: "↓",
    4: "↖", 5: "↗", 6: "↘", 7: "↙"
}

LEFT_TURN: Final[int] = -2

def shift4_tower(direction: Tower, shift: int) -> Tower:
    return cast(Tower, (direction + shift) % NUM_DIRECTIONS_ORTHOGONAL)
def shift4_bishop(direction: Bishop, shift: int) -> Bishop:
    return cast(Bishop, (((direction - NUM_DIRECTIONS_ORTHOGONAL) + shift) % NUM_DIRECTIONS_ORTHOGONAL) + NUM_DIRECTIONS_ORTHOGONAL)

def shift4(direction: King, shift: int) -> King:
    """
    Shifting a direction counter-clockwise, be it a member of Tower or a member of Bishop
    """
    if 0 <= direction < NUM_DIRECTIONS_ORTHOGONAL:
        return shift4_tower(cast(Tower, direction), shift)
    else:
        return shift4_bishop(cast(Bishop, direction), shift)
def shift8(direction: King, shift: int) -> King:
    """
    Shifting the direction counter-clockwise in the group of 8-directions.
    """
    if shift % 2 == 0:
        return shift4(direction, shift//2)
    else:
        direction = shift4(direction, shift//2)
    if 0 <= direction < NUM_DIRECTIONS_ORTHOGONAL:
        return cast(King, direction + NUM_DIRECTIONS_ORTHOGONAL)
    else:
        # 4 -> 1, 5 -> 2, 6 -> 3, 7 -> 0
        return cast(King, (direction - NUM_DIRECTIONS_ORTHOGONAL + 1) % NUM_DIRECTIONS_ORTHOGONAL)
def inverse(direction: King) -> King:
    return shift4(direction, 2)

# Paths:
Path = List[King]
Trans = Tuple[King, Coord] # Transition
FreemanTree = Tuple[Coord, 'FreemanNode']

def path_to_coords(start: Coord, path: Path) -> Tuple[Coords, Coord]:
    """
    Integrate over `path` given the initial position `start`
    """
    coords = set([start])
    coord = start
    for direction in path:
        coord = coord[0] + DIRECTIONS[direction][0], coord[1] + DIRECTIONS[direction][1]
        coords.add(coord)
    return coords, coord

### Freeman

@dataclass
class FreemanNode():
    path: Path
    branches: frozenset['FreemanNode'] # Branches are always separate

    def __str__(self) -> str:
        path = "".join([str(move) for move in self.path])
        if not self.branches:
            return path
        branches = "[" + ", ".join([str(branch)for branch in self.branches]) + "]"
        return path + branches

    def __len__(self) -> int:
        return len(self.path) + sum([len(node) for node in self.branches])

    def __eq__(self, other) -> bool:
        if isinstance(other, 'FreemanNode'):
            return self.path == other.path and self.branches == other.branches
        return False

    def __hash__(self) -> int:
        return hash(self.__repr__())

def arrowify(freeman: FreemanNode) -> None:
    arrows = ""
    for c in str(freeman):
        if c in [str(move) for move in KING]:
            arrows += DIRECTIONS_ARROW[cast(King, int(c))]
        else:
            arrows += c
    print(arrows)
def available_transitions(is_valid: Callable[[Coord], bool], coordinates: Coord) -> List[Trans]:
    transitions = []
    col, row = coordinates
    transitions = []
    for direction in KING:
        ncol, nrow = col + DIRECTIONS[direction][0], row + DIRECTIONS[direction][1]
        ncoordinates = (ncol, nrow)
        if is_valid(ncoordinates):
            transitions.append((direction, ncoordinates))
    return transitions

# Freeman Tree
def encode_connected_component(coords: Coords, start: Coord, is_valid: Callable[[Coord], bool], method="dfs") -> FreemanNode:
    seen = set()

    def coord_to_transitions(coordinates: Coord) -> List[Trans]:
        return available_transitions(is_valid, coordinates)

    def concatenate(moves: Path, node: FreemanNode) -> FreemanNode:
        if not node.branches:
            return FreemanNode(moves + node.path, frozenset())
        elif len(node.branches) == 1:
            child = next(iter(node.branches))
            return FreemanNode(moves + node.path + child.path, child.branches)
        else:
            return FreemanNode(moves + node.path, node.branches)

    def bfs_traversal(start: Coord) -> FreemanNode:
        queue = deque([(start)])
        to_process = []
        seen.add(start)
        branches = set()

        while queue:
            coord = queue.popleft()
            transitions = coord_to_transitions(coord)

            for move, ncoord in transitions:
                if ncoord not in seen:
                    seen.add(ncoord)
                    to_process.append((move, ncoord))

        for move, ncoord in to_process:
            branches.add(concatenate([move], bfs_traversal(ncoord)))
        return concatenate([], FreemanNode([], frozenset(branches)))

    def dfs_traversal(coord: Coord) -> FreemanNode:
        seen.add(coord)
        transitions = coord_to_transitions(coord)
        branches = set()

        for move, ncoord in transitions:
            if ncoord not in seen:
                nnode = concatenate([move], dfs_traversal(ncoord))
                branches.add(nnode)

        return concatenate([], FreemanNode([], frozenset(branches)))

    if method == "bfs":
        return bfs_traversal(start)
    else:
        return dfs_traversal(start)
def decode_freeman(tree: FreemanTree) -> Coords:
    coords = set()
    queue_tree = [tree]

    # Breadth-first collection of coordinates
    # First integrate the main path, then process all branches
    while queue_tree:
        start, freeman = queue_tree.pop()
        ncoords, nstart = path_to_coords(start, freeman.path)
        coords.update(ncoords)
        for branch in freeman.branches:
            queue_tree.append((nstart, branch))
    return coords
#CompressedFreeman = Union[RepeatNode, CompressedNode]

def encode_rl(node: FreemanNode):# -> CompressedNode:
    path = node.path
    segments = []

    if path:
        curr, count = path[0], 1

        for i in range(1, len(path)):
            next = path[i]
            if curr == next:
                count += 1
            else:
                segments.append((curr, count))
                curr, count = next, 1

        segments.append(PatternNode((curr, count), 1))

    return CompressedNode(segments, set(encode_rl(nnode) for nnode in node.branches))

#### Shape tree:

### Real Freeman boundary

def is_boundary(coords: Coords, coord: Coord):
    for i in range(NUM_DIRECTIONS):
        dcol, drow = DIRECTIONS[cast(King, i)]
        ncoord: Coord = (coord[0] + dcol, coord[1] + drow)
        if ncoord not in coords:
            return True
    return False

def mask_to_boundary(coords: Coords) -> Coords:
    boundary: Coords = set()
    for coord in coords:
       if is_boundary(coords, coord):
           boundary.add(coord)
    return boundary

def next_boundary_cell(coords: Coords, current: Coord, dir: King) -> Optional[Tuple[Coord, King]]:
    for i in range(NUM_DIRECTIONS):
        ndir: King = shift8(dir, LEFT_TURN + i)
        dcol, drow = DIRECTIONS[ndir]
        ncoord: Coord = (current[0] + dcol, current[1] + drow)
        if ncoord in coords and is_boundary(coords, ncoord):
            return ncoord, ndir
    return None

def trace_boundary(coords: Coords, start: Coord) -> Path:
    boundary: Path = []
    current = start
    dir: King = 1  # Start with direction right

    while (current != start or not boundary):
        ncell = next_boundary_cell(coords, current, dir)

        match ncell:
            case None:
                break
            case (ncoord, ndir):
                boundary.append(ndir)
                current = ncoord
                dir = ndir


    return boundary

def freeman_to_boundary_coords(start: Coord, freeman: Path) -> Coords:
    """Integrate tge freeman chain code"""
    coords = set([start])
    current = start
    for dir in freeman:
        dcol, drow = DIRECTIONS[dir]
        current = current[0] + dcol, current[1] + drow
        coords.add(current)
    return coords
