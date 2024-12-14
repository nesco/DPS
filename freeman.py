
from typing import Optional, Callable, Literal, Final
from typing import cast

from collections import deque

from helpers import TraversalModes
from helpers import *
from lattice import *
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
    0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1),
    4: (-1, -1), 5: (1, -1), 6: (1, 1), 7: (-1, 1)
}

DIRECTIONS_ARROW: Final[dict[King, str]] = {
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
Path = list[King]
Trans = tuple[King, Coord] # Transition
FreemanTree = tuple[Coord, 'FreemanNode']

def path_to_coords(start: Coord, path: Path) -> tuple[Coords, Coord]:
    """
    Integrate over `path` given the initial position `start`
    """
    coords = set([start])
    coord = start
    for direction in path:
        coord = coord[0] + DIRECTIONS_FREEMAN[direction][0], coord[1] + DIRECTIONS_FREEMAN[direction][1]
        coords.add(coord)
    return coords, coord

### Freeman
# A Freeman tree is a trie on Freeman chain codes describing pathes
# Contrarily to boundaries, several pathes are required to describe shapes
# because of topological issues: filling a shape with a path can "disconnect" remaining
# areas to explore into separated connected components

@dataclass(frozen=True)
class FreemanNode():
    """Represents a node in the Freeman chain code trie"""
    path: Path
    children: list['FreemanNode'] # Branches are always separate


    def __len__(self) -> int:
        return len(self.path) + sum([len(child) for child in self.children])

    def __str__(self) -> str:
        path_str = "".join([str(move) for move in self.path])
        if not self.children:
            return path_str
        children_str = "[" + ", ".join([str(child) for child
            in self.children]) + "]"
        return f"{path_str}{children_str}"

    def __eq__(self, other) -> bool:
        if isinstance(other, 'FreemanNode'):
            return self.path == other.path and self.children == other.children
        return False

class Freeman:
    """Freeman chain code processor"""
    DIRECTIONS: Final[dict[King, Coord]] = {
        0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1),
        4: (-1, -1), 5: (1, -1), 6: (1, 1), 7: (-1, 1)
    }

    def __init__(self, is_valid_coord: Callable):
        self.is_valid = is_valid_coord
        self.seen: set[Coord] = set()

    def get_valid_moves(self, coord: Coord) -> list[tuple[King, Coord]]:
        """Get the list of valid moves from the current coordinates"""
        moves = []
        for direction in range(8):
            dir_cast = cast(King, direction)
            dx, dy = self.DIRECTIONS[dir_cast]
            ncoord  = (coord[0] + dx, coord[1] + dy)
            if self.is_valid(ncoord):
                moves.append((dir_cast, ncoord))
        return moves

    def build_trie(self, start: Coord, mode: TraversalModes = TraversalModes.DFS) -> FreemanNode:
        """Build a Freeman chain code tree from a start coordinate"""
        self.seen = {start}
        if mode == TraversalModes.DFS:
            return self._dfs_trie(start)
        return self._bfs_trie(start)

    def _dfs_trie(self, coord: Coord) -> FreemanNode:
        """Build trie using depth-first search"""
        children = []

        for direction, next_coord in self.get_valid_moves(coord):
               if next_coord not in self.seen:
                   self.seen.add(next_coord)
                   child_node = self._dfs_trie(next_coord)
                   if child_node.path or child_node.children:
                       path = cast(Path, [direction] + child_node.path)
                       if len(child_node.children) == 1:
                           # Linearize by combining path with single child
                           child = child_node.children[0]
                           children.append(FreemanNode(path + child.path, child.children))
                       else:
                           children.append(FreemanNode(path, child_node.children))

        return FreemanNode([], children)

    def _bfs_trie(self, start: Coord) -> FreemanNode:
        """Build a trie using breadth-first search"""
        queue = deque([(start, [])])
        children = []

        while queue:
            coord, path = queue.popleft()

            for direction, next_coord in self.get_valid_moves(coord):
                if next_coord not in self.seen:
                    self.seen.add(next_coord)
                    npath = path + [direction]
                    queue.append((next_coord, npath))
                    children.append(FreemanNode(npath, []))

        # Post-process to linearize nodes with single children
        nchildren = []
        for child in children:
            if len(child.children) == 1:
                grandchild = child.children[0]
                nchildren.append(FreemanNode(child.path + grandchild.path, grandchild.children))
            else:
                nchildren.append(child)

        return FreemanNode([], nchildren)

def decode_freeman(start: Coord, root: FreemanNode):
    """Convert Freeman trie back to coordinates"""
    coords = {start}
    queue = [(start, root)]

    while queue:
        current, node = queue.pop()
        for child in node.children:
            coord = current
            for direction in child.path:
                dx, dy = Freeman.DIRECTIONS[direction]
                coord = (coord[0] + dx, coord[1] + dy)
                coords.add(coord)
            queue.append((coord, child))

    return coords

def arrowify(root: FreemanNode):
    """Convert Freeman chain code to directional arrows."""
    def convert_char(c: str) -> str:
        return DIRECTIONS_ARROW[cast(King, int(c))] if c.isdigit() else c
    result = "".join(map(convert_char, str(root)))
    print(result)

def arrowify1(freeman: FreemanNode) -> None:
    result = "".join(DIRECTIONS_ARROW[cast(King, int(c))] if c.isdigit() else c
                    for c in str(freeman))
    print(result)
def available_transitions_freeman(is_valid: Callable[[Coord], bool], coordinates: Coord) -> list[Trans]:
    transitions = []
    col, row = coordinates
    transitions = []
    for direction in KING:
        ncol, nrow = col + DIRECTIONS_FREEMAN[direction][0], row + DIRECTIONS_FREEMAN[direction][1]
        ncoordinates = (ncol, nrow)
        if is_valid(ncoordinates):
            transitions.append((direction, ncoordinates))
    return transitions

# Freeman Tree
def encode_connected_component(start: Coord, is_valid: Callable[[Coord], bool], method: TraversalModes = TraversalModes.DFS) -> FreemanNode:
    seen = set()

    def coord_to_transitions(coordinates: Coord) -> list[Trans]:
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
def decode_freeman1(tree: FreemanTree) -> Coords:
    coords = set()
    queue_tree = [tree]

    # Breadth-first collection of coordinates
    # First integrate the main path, then process all branches
    while queue_tree:
        start, freeman = queue_tree.pop()
        ncoords, nstart = path_to_coords(start, freeman.path)
        coords.update(ncoords)
        for child in freeman.children:
            queue_tree.append((nstart, child))
    return coords

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

def next_boundary_cell(coords: Coords, current: Coord, dir: King) -> Optional[tuple[Coord, King]]:
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
        dcol, drow = DIRECTIONS_FREEMAN[dir]
        current = current[0] + dcol, current[1] + drow
        coords.add(current)
    return coords
