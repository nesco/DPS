
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

TOWER: Final[List[Tower]] = [0, 1, 2, 3]
BISHOP: Final[List[Bishop]] = [4, 5, 6, 7]
KING: Final[List[King]] = [0, 1, 2, 3, 4, 5, 6, 7]

DIRECTIONS: Final[Dict[King, Coord]] = {
    0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1),
    4: (-1, -1), 5: (1, -1), 6: (1, 1), 7: (-1, 1)
}

DIRECTIONS_ARROW: Final[Dict[King, str]] = {
    0: "←", 1: "↑", 2: "→", 3: "↓",
    4: "↖", 5: "↙", 6: "↘", 7: "↗"
}

def shift4_tower(direction: Tower, shift: int) -> Tower:
    return cast(Tower, (direction + shift) % 4)
def shift4_bishop(direction: Bishop, shift: int) -> Bishop:
    return cast(Bishop, (((direction - 4) + shift) % 4) + 4)

def shift4(direction: King, shift: int) -> King:
    """
    Shifting a direction counter-clockwise, be it a member of Tower or a member of Bishop
    """
    if 0 <= direction < 4:
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
    if 0 <= direction < 4:
        return cast(King, direction + 4)
    else:
        # 4 -> 1, 5 -> 2, 6 -> 3, 7 -> 0
        return cast(King, (direction - 3) % 4)
def inverse(direction: King) -> King:
    return shift4(direction, 2)

# Paths:
Path = List[King]

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
Trans = Tuple[King, Coord] # Transition

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
    row, col = coordinates
    transitions = []
    for direction in KING:
        nrow, ncol = row + DIRECTIONS[direction][0], col + DIRECTIONS[direction][1]
        ncoordinates = (nrow, ncol)
        if is_valid(ncoordinates):
            transitions.append((direction, ncoordinates))
    return transitions

# Freeman Tree
FreemanTree = Tuple[Coord, FreemanNode]
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
