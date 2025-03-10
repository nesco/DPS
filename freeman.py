from typing import Optional, Callable, Literal, Final
from typing import cast

from collections import deque

from helpers import TraversalModes
from helpers import *

# from lattice import *
from dataclasses import dataclass

# Directions

Tower = Literal[0, 1, 2, 3]
Bishop = Literal[4, 5, 6, 7]
King = Tower | Bishop

NUM_DIRECTIONS: Final[int] = 8
NUM_DIRECTIONS_ORTHOGONAL: Final[int] = 4

TOWER: Final[list[Tower]] = [0, 1, 2, 3]
BISHOP: Final[list[Bishop]] = [4, 5, 6, 7]
KING: Final[list[King]] = [0, 1, 2, 3, 4, 5, 6, 7]

DIRECTIONS_FREEMAN: Final[dict[King, Coord]] = {
    0: (-1, 0),
    1: (0, -1),
    2: (1, 0),
    3: (0, 1),
    4: (-1, -1),
    5: (1, -1),
    6: (1, 1),
    7: (-1, 1),
}

DIRECTIONS_ARROW: Final[dict[King, str]] = {
    0: "←",
    1: "↑",
    2: "→",
    3: "↓",
    4: "↖",
    5: "↗",
    6: "↘",
    7: "↙",
}

LEFT_TURN: Final[int] = -2


def shift4_tower(direction: Tower, shift: int) -> Tower:
    return cast(Tower, (direction + shift) % NUM_DIRECTIONS_ORTHOGONAL)


def shift4_bishop(direction: Bishop, shift: int) -> Bishop:
    return cast(
        Bishop,
        (((direction - NUM_DIRECTIONS_ORTHOGONAL) + shift) % NUM_DIRECTIONS_ORTHOGONAL)
        + NUM_DIRECTIONS_ORTHOGONAL,
    )


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
        return shift4(direction, shift // 2)
    else:
        direction = shift4(direction, shift // 2)
    if 0 <= direction < NUM_DIRECTIONS_ORTHOGONAL:
        return cast(King, direction + NUM_DIRECTIONS_ORTHOGONAL)
    else:
        # 4 -> 1, 5 -> 2, 6 -> 3, 7 -> 0
        return cast(
            King,
            (direction - NUM_DIRECTIONS_ORTHOGONAL + 1) % NUM_DIRECTIONS_ORTHOGONAL,
        )


def inverse(direction: King) -> King:
    return shift4(direction, 2)


# Paths:
Path = list[King]
Transition = tuple[King, Coord]  # Transition
FreemanTree = tuple[Coord, "FreemanNode"]


### Freeman
# A Freeman tree is a trie on Freeman chain codes describing pathes
# Contrarily to boundaries, several pathes are required to describe shapes
# because of topological issues: filling a shape with a path can "disconnect" remaining
# areas to explore into separated connected components


@dataclass(frozen=True)
class FreemanNode:
    """Represents a node in the Freeman chain code trie"""

    path: Path
    children: list["FreemanNode"]  # Branches are always separate

    def __len__(self) -> int:
        return len(self.path) + sum([len(child) for child in self.children])

    def __str__(self) -> str:
        path_str = "".join([str(move) for move in self.path])
        if not self.children:
            return path_str
        children_str = "[" + ", ".join([str(child) for child in self.children]) + "]"
        return f"{path_str}{children_str}"

    def __eq__(self, other) -> bool:
        if isinstance(other, "FreemanNode"):
            return self.path == other.path and self.children == other.children
        return False

def decode_freeman(start: Coord, root: FreemanNode) -> Coords:
    """Convert Freeman trie back to coordinates"""
    coords = {start}
    queue = [(start, root)]

    while queue:
        current, node = queue.pop()
        for child in node.children:
            coord = current
            for direction in child.path:
                dx, dy = DIRECTIONS_FREEMAN[direction]
                coord = (coord[0] + dx, coord[1] + dy)
                coords.add(coord)
            queue.append((coord, child))

    return coords


def arrowify(root: FreemanNode):
    """Convert Freeman chain code to directional arrows."""

    def movement_to_arrow(c: str) -> str:
        return DIRECTIONS_ARROW[cast(King, int(c))] if c.isdigit() else c

    result = "".join(map(movement_to_arrow, str(root)))
    print(result)

def available_transitions_freeman(
    is_valid: Callable[[Coord], bool], coordinates: Coord
) -> list[Transition]:
    transitions = []
    col, row = coordinates
    transitions = []
    for direction in KING:
        ncol, nrow = (
            col + DIRECTIONS_FREEMAN[direction][0],
            row + DIRECTIONS_FREEMAN[direction][1],
        )
        ncoordinates = (ncol, nrow)
        if is_valid(ncoordinates):
            transitions.append((direction, ncoordinates))
    return transitions


# Freeman Tree
def encode_connected_component(
    start: Coord,
    is_valid: Callable[[Coord], bool],
    method: TraversalModes = TraversalModes.DFS,
) -> FreemanNode:
    seen = set()

    def coord_to_transitions(coordinates: Coord) -> list[Transition]:
        return available_transitions_freeman(is_valid, coordinates)

    def concatenate(moves: Path, node: FreemanNode) -> FreemanNode:
        if not node.children:
            return FreemanNode(moves + node.path, list())
        elif len(node.children) == 1:
            child = next(iter(node.children))
            return FreemanNode(moves + node.path + child.path, child.children)
        else:
            return FreemanNode(moves + node.path, node.children)

    def bfs_traversal(start: Coord) -> FreemanNode:
        queue = deque([(start)])
        to_process = []
        seen.add(start)
        branches = list()

        while queue:
            coord = queue.popleft()
            transitions = coord_to_transitions(coord)

            for move, ncoord in transitions:
                if ncoord not in seen:
                    seen.add(ncoord)
                    to_process.append((move, ncoord))

        for move, ncoord in to_process:
            branches.append(concatenate([move], bfs_traversal(ncoord)))
        return concatenate([], FreemanNode([], list(branches)))

    def dfs_traversal(coord: Coord) -> FreemanNode:
        seen.add(coord)
        transitions = coord_to_transitions(coord)
        branches = list()

        for move, ncoord in transitions:
            if ncoord not in seen:
                nnode = concatenate([move], dfs_traversal(ncoord))
                branches.append(nnode)

        return concatenate([], FreemanNode([], list(branches)))

    if method == TraversalModes.BFS:
        return bfs_traversal(start)
    else:
        return dfs_traversal(start)

### Real Freeman boundary


def is_boundary(coords: Coords, coord: Coord):
    for i in range(NUM_DIRECTIONS):
        dcol, drow = DIRECTIONS_FREEMAN[cast(King, i)]
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


def next_boundary_cell(
    coords: Coords, current: Coord, dir: King
) -> Optional[tuple[Coord, King]]:
    for i in range(NUM_DIRECTIONS):
        ndir: King = shift8(dir, LEFT_TURN + i)
        dcol, drow = DIRECTIONS_FREEMAN[ndir]
        ncoord: Coord = (current[0] + dcol, current[1] + drow)
        if ncoord in coords and is_boundary(coords, ncoord):
            return ncoord, ndir
    return None


def trace_boundary(coords: Coords, start: Coord) -> Path:
    boundary: Path = []
    current = start
    dir: King = 1  # Start with direction right

    while current != start or not boundary:
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
        dcol, drow = DIRECTIONS_FREEMAN[dir]
        current = current[0] + dcol, current[1] + drow
        coords.add(current)
    return coords
