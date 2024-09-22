"""
Grid can be totally or partially encoded in a lossless fashion into Asbract Syntax Trees.
The lossless part is essential here because ARC Corpus sets the problem into the low data regime.
First grids are marginalized into connected components of N colors.
Those connected components are then represented through their graph traversals using branching freeman chain codes.

The lossless encoding used is basic pattern matching for (meta) repetitions through Repeat and SymbolicNode.
The language formed by ASTs are simple enough an approximate version of kolmogorov complexity can be computed.
It helps choosing the most efficient encoding, which is the closest thing to a objective  proper representation
of the morphology.

# Naming conventions
# "nvar" == "var_new"
# "cvar" == "var_copy"
"""
DEBUG_RESOLVE = False
DEBUG_UNSYMBOLIZE = False
DEBUG_NODELIST = False
DEBUG_ROOT = True
DEBUG = False
DEBUG_ASTMAP = False

LEN_COORD = 10
LEN_COLOR = 4
LEN_NODE = 3 # Base length for node type (3 bits) because <= 8 types
LEN_MOVE = 3 # Assuming 8-coinnectivity 3bits per move
LEN_COUNT = 4   # counts of repeats should be an int between 2 and 9 or -2 and -8 (4 bits) ideally
LEN_INDEX = 3 # should not be more than 8 so 3 bits
LEN_INDEX_VARIABLE = 1 # Variable can be 0 or 1 so 1 bit
LEN_RECT = 8

from dataclasses import dataclass, asdict
from collections import deque

from math import ceil

from typing import Any, List, Union, Optional, Iterator, Callable, Set, Tuple, Dict, NewType, cast
from helpers import *

from time import sleep
from freeman import shift4, inverse

#### Abstract Syntax Tree Structure

#directions = {
#    '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
#    '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
#}


@dataclass
class Node:
    """
    Abstract Node class used to stuff inherited methods.
    """

    def __len__(self) -> int:
           return LEN_NODE  # Base length for node type (3 bits) because <= 8 types

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__class__.__name__ == other.__class__.__name__ and hash(self) == hash(other)

    def __repr__(self) -> str:
            return str(asdict(self))

    def __hash__(self) -> int:
        return hash(repr(self))

    def __add__(self, other) -> 'ASTNode':
        if isinstance(other, NodeList):
            return NodeList([self] + other.nodes)
        elif isinstance(other, (Variable, Root)):
            return NotImplemented
        elif issubclass(other.__class__, Node):
            return NodeList([self, other])
        else:
            return NotImplemented

    def depth_iter(self) -> Iterator['ASTNode']:
        return iter(())

    def breadth_iter(self) -> Iterator['ASTNode']:
        return iter(())

    def copy(self) -> 'ASTNode':
        return Node()
    # Return the node in the reverse order
    def reverse(self) -> 'ASTNode':
        return self

@dataclass
class Moves(Node):
    """
    Moves store litteral non branching parts of a Freeman code Chain.
    It's a container for a string of octal digits, each indicating a direction in 8-cconectivity.
    The directions are defined by:
        directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
        }
    """

    moves: str  #0, 1, 2, 3, 4, 5, 6, 7 (3 bits)

    def simplify_repetitions(self):
        moves_ls = [Moves(k) for k in self.moves]
        simplified = []
        i = 0
        while i < len(moves_ls):
            pattern, count, reverse = find_repeating_pattern(moves_ls[i:], 0)
            if pattern is None or count <= 1:
                # No repeating pattern found, add the current move and continue
                simplified.append(moves_ls[i])
                i += 1
            else:
                if len(pattern) == 1:
                    node = pattern[0]
                else:
                    node = Moves(''.join(m.moves for m in pattern))
                if reverse:
                    simplified.append(Repeat(node, -count))
                else:
                    simplified.append(Repeat(node, count))
                i += len(pattern) * count

        if len(simplified) == 1:
            return simplified[0]
        else:
            result = []
            curr = simplified[0]
            for i in range(1, len(simplified)):
                next_node = simplified[i]
                if isinstance(curr, Moves) and isinstance(next_node, Moves):
                    curr = Moves(curr.moves + next_node.moves)
                else:
                    result.append(curr)
                    curr = next_node
            result.append(curr)

            # If after concatenation we're left with a single Moves object
            # that's identical to self, return self
            if len(result) == 1 and isinstance(result[0], Moves) and result[0].moves == self.moves:
                return self

            return NodeList(result) if len(result) > 1 else result[0]

    def __len__(self) -> int:
        return super().__len__() + LEN_MOVE * len(self.moves) # Assuming 8-coinnectivity 3bits per move
    def __str__(self) -> str:
        return self.moves
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def __add__(self, other): # right addition
        match other:
            case str():
                return Moves(self.moves + other)
            case Moves():
                return encode_run_length(Moves(self.moves + other.moves))
            case _:
                return super().__add__(other)
    def __iter__(self) -> Iterator[str]:
        return iter(self.moves)
    def iter_path(self, start) -> Iterator[tuple]:
        """
        Iterator over the path
        """
        col, row = start
        for move in self.moves:
            col += DIRECTIONS[move][0]
            row += DIRECTIONS[move][1]
            yield (col, row)
    def copy(self) -> 'ASTNode':
        return Moves(self.moves)
    def reverse(self) -> 'ASTNode':
        return Moves(self.moves[::-1])

@dataclass
class Rect(Node):
    height: Union[int, 'Variable']
    width: Union[int, 'Variable']

    def __len__(self) -> int:
        return super().__len__() + LEN_RECT
    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def __eq__(self, other) -> bool:
        return isinstance(other, Rect) and other.height == self.height and other.width == self.width
    def copy(self) -> 'ASTNode':
        return Rect(self.height, self.width)

@dataclass
class Repeat(Node):
    node: 'ASTNode'
    count: Union[int, 'Variable']  # should be an int between 2 and 17 (4 bits) ideally

    def __len__(self) -> int:
        return super().__len__() + len(self.node) + LEN_COUNT
    def __eq__(self, other) -> bool:
        match other:
            case Repeat(node=n, count=c):
                 return (self.node == n and self.count == c)
            case _:
                return False
    def __str__(self):
            return f"({str(self.node)})*{{{self.count}}}"
    def __iter__(self) -> Iterator['ASTNode']:
        if self.node is not None:
            if isinstance(self.count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")
            # Handle negative values as looping
            elif self.count < 0:
                return (self.node if i%2==0 else self.node.reverse() for i in range(-self.count))
            else:
                return (self.node for _ in range(self.count))
        return iter(())
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def __add__(self, other):
        match (self, other):
            case (Repeat(node=n1, count=c1), Repeat(node=n2, count=c2)) if n1==n2:
                return Repeat(node=n1, count=c1+c2)
            case (Repeat(node=n1, count=c1), _) if n1 == other:
                return Repeat(node=n1, count=c1+1)
            case _:
                return super().__add__(other)
    def depth_iter(self) -> Iterator['ASTNode']:
        yield self.node
        if isinstance(self.node, (Branch, Repeat, NodeList)):
            yield from self.node.depth_iter()
    def breadth_iter(self) -> Iterator['ASTNode']:
        yield self.node
        if isinstance(self.node, (Branch, Repeat, NodeList)):
            yield from self.node.breadth_iter()
    def copy(self):
        return Repeat(self.node.copy(), self.count)
    def reverse(self):
        return Repeat(self.node.reverse(), self.count)

class NodeList:
    """
    NodeList is a container representing a possibly Branching Freeman Code chain
    where subparts are encoded AST Nodes.
    As an abstract container designed to replace List() with an object having the right methods,
    it doesn't count for the length.
    """
    def __init__(self, nodes: List['ASTNode']):
        #if DEBUG_NODELIST:
        #    print(f"Trying to initialize a NodeList with nodes:")
        #for i, n in enumerate(nodes):
            #if DEBUG_NODELIST:
            #    print(f"Node n°{i}: {n}")
        # Concatening Moves
        self.nodes = []
        current, nnodes = nodes[0], nodes[1:]
        while nnodes:
            next, nnodes = nnodes[0], nnodes[1:]

            ## Structural Pattern Matching
            match (current, next):
                case (Root(), _):
                    raise NotImplementedError("Trying to initialise a node list with : " + ','.join(str(n) for n in nodes))
                case (Moves(), Moves()): # Concatening Moves
                    current += next
                case (_, Repeat(node=n1, count=c1)) if current == n1:
                    current = Repeat(n1, c1+1)
                case (NodeList(), _): #Flattening NodeList
                    self.nodes.extend(current)
                    current = next
                case (Repeat(node=n1, count=c1), Repeat(node=n2, count=c2)) if n1 == n2 and isinstance(c1, int) and isinstance(c2, int):  #Concatening Repeats
                    current.count += c2
                # There can be nothing after a Branch, because it becomes implicitely a branch
                case (Branch(), _):
                    while nnodes:
                        if isinstance(next, Branch):
                            current.sequences.extend(next.sequences)
                            next, nnodes = nnodes[0], nnodes[1:]
                        else:
                            current.sequences.append(NodeList([next] + nnodes))
                            nnodes = None
                    break  # Exit the main loop after processing the Branch (even though it should exit already)
                case _:
                    self.nodes.append(current)
                    current = next
        if isinstance(current, Moves):
            current = encode_run_length(current)
        if isinstance(current, NodeList):
            self.nodes.extend(current)
        else:
            self.nodes.append(current)

    def __len__(self) -> int:
        return sum(len(node) for node in self.nodes) # Sequence doesn't add bits
    def __str__(self) -> str:
        return "".join(str(node) for node in self.nodes)
    def __repr__(self) -> str:
        return self.__class__.__name__ + "(nodes=" + self.nodes.__repr__() +")"
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def __eq__(self, other) -> bool:
        if not isinstance(other, NodeList):
            return False
        return len(self.nodes) == len(other.nodes) and all(
            s1 == s2 for s1, s2 in zip(self.nodes, other.nodes)
        )
    def __add__(self, other):
        if isinstance(other, NodeList):
            return NodeList(self.nodes + other.nodes)
        elif isinstance(other, str):
            return self.__add__(Moves(other))
        elif isinstance(other, Moves):
            if self.nodes and isinstance(self.nodes[-1], Node):
                nnodes = self.nodes[:-1] + [self.nodes[-1]+other]
            else:
                nnodes = self.nodes + [other]
            return NodeList(nnodes)
        elif isinstance(other, Repeat):
            if self.nodes and isinstance(self.nodes[-1], Repeat) and \
            self.nodes[-1].node == other.node:
                nnodes = self.nodes[:-1] + [Repeat(other.node, self.nodes[-1].count + other.count)]
                return NodeList(nnodes)
            else:
                return NodeList(self.nodes + [other])
        elif isinstance(other, Branch):
            return NodeList(self.nodes + [other])
        else:
            return NotImplemented
    def __iter__(self) -> Iterator['ASTNode']:
        return (node for node in self.nodes if node is not None)
    def depth_iter(self) -> Iterator['ASTNode']:
        for node in self.nodes:
            yield node
            if isinstance(node, (Branch, Repeat, NodeList)):
                yield from node.depth_iter()
    def breadth_iter(self) -> Iterator['ASTNode']:
        iterators = []
        for node in self.nodes:
            yield node
            if isinstance(node, (Branch, Repeat, NodeList)):
                iterators.append(node.breadth_iter())
        for it in iterators:
            yield from it

    def copy(self):
        return NodeList([node.copy() for node in self.nodes])
    def reverse(self) -> 'ASTNode':
        reverse_list = self.nodes[::-1]
        nnodes = [node.reverse() for node in reverse_list]
        return NodeList(nnodes)

class Branch(Node):
    """
    Represents branching of the traversal of a connected component.
    As a single freeman code chain can't represent all connected components,
    it needs to have branching parts. This node encode the branching.

    As a branch is never a single node, it can be used to encode iterators through a Repeat.
    If a Branch contains a single repeat, then it will act like a positive or negative iterator
    """
    def __init__(self, sequences: List['ASTNode']):
        self.sequences = []
        # Done to better handle crosses like : (2, 2):[00,11,22,33]
        for seq in sequences:
            curr = seq
            if isinstance(curr, Moves):
                curr = encode_run_length(curr)
            self.sequences.append(curr)
    def check_iterators(self):
        # if the sequences can be represented as a Repeat repurposed as an iterator
        self.sequences = get_iterator(self.sequences)
    def __len__(self) -> int:
        return super().__len__() + sum(len(sequence) for sequence in self.sequences)
    def __str__(self) -> str:
        if len(self.sequences) == 1 and isinstance(self.sequences[0], Repeat):
            return "[+" + str(self.sequences[0]) + "]"
        return "[" + ",".join(str(sequence) for sequence in self.sequences) +"]"
    def __eq__(self, other) -> bool:
        if not isinstance(other, Branch):
            return False
        return len(self.sequences) == len(other.sequences) and all(
            s1 == s2 for s1, s2 in zip(self.sequences, other.sequences)
        )
    def __add__(self, other):
        if isinstance(other, Branch):
            return Branch(self.sequences+other.sequences)
        else:
            return super().__add__(other)
    def __iter__(self) -> Iterator['ASTNode']:
        # It needs to handle the repeat used as an iterator case
        if len(self.sequences) == 1 and isinstance(self.sequences[0], Repeat):
            n = self.sequences[0].node
            count = self.sequences[0].count

            if isinstance(count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")

            i=1
            if isinstance(count, int) and count < 0:
                i=-1
                count = -count
            return (shift_moves(k*i, n) for k in range(count))
        return iter(self.sequences)
    def __repr__(self) -> str:
        return f"Branch(sequences={self.sequences!r})"
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def depth_iter(self) -> Iterator['ASTNode']:
        for node in self.sequences:
            yield node
            if isinstance(node, (Branch, Repeat, NodeList)):
                yield from node.depth_iter()
    def breadth_iter(self) -> Iterator['ASTNode']:
        iterators = []
        for node in self.sequences:
            yield node
            if isinstance(node, (Branch, Repeat, NodeList)):
                iterators.append(node.breadth_iter())
        for it in iterators:
            yield from it
    def copy(self):
        return Branch([ seq.copy() for seq in self.sequences])
    def reverse(self) -> 'ASTNode':
        reverse_list = self.sequences[::-1]
        nsequences = [node.reverse() for node in reverse_list]
        return Branch(nsequences)

@dataclass
class Variable(Node):
    """
    This Node serves to reminds where SymbolicNodes parameters need to be pasted.
    """
    index: int # The hash of the replacement pattern to know it will replace this

    def __len__(self):
        return super().__len__() + LEN_INDEX_VARIABLE

    def __str__(self):
        return f"Var({self.index})"

    def __repr__(self):
        return self.__class__.__name__

    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        return isinstance(other, Variable)
    def copy(self):
        return Variable(self.index)

@dataclass
class SymbolicNode(Node):
    """
    To make the AST representations of branching code chains powerful,
    it needs to be able to compress efficiently code chains,
    with the possibility to memorize reoccuring patterns,
    be their constants or abstracted as functions of single parameters.
    The pattern needs to be stored independently in a list of patterns, index is the index of the pattern in this list
    """
    index: int # should not be more than 8 so 3 bits
    param: Any # for repeats
    len_ref: int # len of the ref

    def __len__(self) -> int:
        return  super().__len__() + LEN_INDEX + len_param(self.param) # Not the super, it's a placeholder for something that already counts the +2
    def __str__(self) -> str:
        if self.param:
            return f" s_{self.index}({self.param}) "
        return f" s_{self.index} "
    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __reverse(self) -> 'ASTNode':
        param = self.param
        if isinstance(param, ASTNode):
            param = param.reverse()
        return SymbolicNode(self.index, param, self.len_ref)

    def copy(self):
        return SymbolicNode(self.index, self.param.copy() if isinstance(self.param, ASTNode) else self.param, self.len_ref)

@dataclass
class BiSymbolicNode(Node):
    """
    Test to enable products, lots of things are product, particularly squares
    """
    index: int
    param1: Any
    param2: Any
    len_ref: int

    def __len__(self) -> int:
        return  super().__len__() + LEN_INDEX + len_param(self.param1) + len_param(self.param2)

    def __str__(self) -> str:
        return f" s_{self.index}({self.param1}, {self.param2}) "
    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __reverse(self) -> 'ASTNode':
        param1 = self.param1
        if isinstance(param1, ASTNode):
            param1 = param1.reverse()
        param2 = self.param2
        if isinstance(param1, ASTNode):
            param2 = param2.reverse()
        return BiSymbolicNode(self.index, param1, param2, self.len_ref)

    def copy(self):
        return BiSymbolicNode(self.index, self.param1.copy() if isinstance(self.param1, ASTNode) else self.param1 \
            , self.param2.copy() if isinstance(self.param2, ASTNode) else self.param2, self.len_ref)

@dataclass
class Root(Node):
    """
    Node representing a path root. Note that the branches here can possibly lead to overlapping paths
    """
    start: Union[Coord, Variable]
    colors: Union[Colors, Variable]
    node: Optional['ASTNode']

    def __post_init_(self):
        if DEBUG_ROOT:
            match self.node:
                case Root(_):
                    print(f"Trying to initialize {self.__repr__()} with another root")
                    print(f"Root: {self.node}")
                case SymbolicNode(i, p, _) if isinstance(p, Root):
                    print(f"Trying to initialize {self.__repr__()} with another root")
                    print(f"Root: {p}")
            col, row = self.start
            if col < 0 or row < 0:
                print(f"Trying to initialize {self} at nehative starting point {col}, {row}")
    def __len__(self):
        # 10 bits for x and y going from 0 to 32 max on the grids + 4 bits for each color (10 choices)
        len_node = len(self.node) if self.node is not None else 0
        return LEN_COORD + len(self.colors) * LEN_COLOR + len_node
    def __add__(self, other):
        match other:
            case Variable():
                return NotImplemented
            case Root(start=(col, row), colors=c, child=n) if not isinstance(self.start, Variable):
                if self.start[0] == col and self.start[1] == row:
                    if self.colors != c or isinstance(self.colors, Variable) or isinstance(c, Variable):
                        raise NotImplementedError()
                    return Root((col, row), c, Branch([self.node, n]))
                return Branch([self, other])
            case Root(start=s, colors=c, child=n) if  (isinstance(self.start, Variable) or isinstance(s, Variable)):
                raise NotImplementedError()
        return Root(self.start, self.colors, Branch([self.node, other]))
    def __str__(self):
        return f"{self.colors}->{self.start}:" + str(self.node)
    #def __repr__(self):
    #    return f"{self.__class__.__name__}(start={self.start.__repr__()}, colors={str(self.colors)}, node={self.node.__repr__()})"
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        if not isinstance(other, Root):
            return False
        #if isinstance(self.start, Variable):
        #    return (self.colors==other.colors) and (self.node == other.node)
        return (self.start == other.start) and (self.colors==other.colors) \
        and (self.node == other.node)

    def depth_iter(self) -> Iterator['ASTNode']:
        if self.node is None:
            return iter(())
        yield self.node
        if isinstance(self.node, (Branch, Repeat, NodeList)):
            yield from self.node.depth_iter()

    def breadth_iter(self) -> Iterator['ASTNode']:
        if self.node is None:
            return iter(())
        yield self.node
        if isinstance(self.node, (Branch, Repeat, NodeList)):
            yield from self.node.breadth_iter()

    def copy(self):
        cnode = self.node.copy() if self.node is not None else None
        return Root(self.start, self.colors, cnode)

    def reverse(self):
        node = self.node.reverse() if self.node is not None else None
        return Root(self.start, self.colors, node)

# Strategy diviser pour régner avec marginalisation + reconstruction

@dataclass
class UnionNode(Node):
    """
    Represent a connected component by reconstructing it with the best set of single color programs.
    After marginalisation comes reconstruction, divide and conquer.
    """
    #background: Dict['ASTNode', 'ASTNode']
    codes: Set['ASTNode']
    shadowed: Optional[Set[int]] = None
    background: Optional['ASTNode'] = None

    def __len__(self):
        len_codes = 0
        if self.codes is None:
            return 0
        return sum([len(code) for code in self.codes])
    def __add__(self, other):
        raise NotImplemented
    def __str__(self):
        msg = ''
        if self.background is not None:
            msg += f'{self.background } < '
        if self.codes is None:
            msg += 'Ø'
        else:
            msg += 'U'.join([f"{{{code}}}" for code in self.codes])
        return msg
    def __repr__(self):
        codes_repr = ','.join(code.__repr__() for code in self.codes)
        return f"{self.__class__.__name__}(codes={codes_repr})"
    def __eq__(self, other) -> bool:
        if not isinstance(other, UnionNode):
            return False
        return (self.background == other.background and self.codes == other.codes)
    def __hash__(self):
        return hash(self.__repr__())
############## TEST : NEW FREEMAN STRUCTURE ###########

Segment = Tuple[str, int]

@dataclass
class PatternNode():
    patterns: List[Union['PatternNode', Segment]]
    count: int

    def __len__(self) -> int:
        return 3 + sum(len(pattern) for pattern in self.patterns)

    def __eq__(self, other) -> bool:
        return self.count == other.count and self.patterns == other.patterns

@dataclass
class CompressedNode():
    path: List[PatternNode]
    branches: Set['CompressedFreeman']

    def __len__(self):
        # 3 bits for the move
        # 4 for the count (16 rep max?)
        return sum([3*4 for _ in self.path]) + sum([len(node) for node in self.branches])

    def __hash__(self) -> int:
        return hash(self.__repr__())

@dataclass
class RepeatNode():
    node: 'CompressedFreeman'
    count: int

    def __eq__(self, other) -> bool:
        match other:
            case RepeatNode(node, count):
                return self.node == node and self.count == count
            case CompressedNode():
                return False
        return False

    def __iter__(self):
        pass
    def __len__(self):
        return self.count * len(self.node) if self.count > 0 else -self.count*len(self.node)

#CompressedFreeman = Union[RepeatNode, CompressedNode]
############
## Types
ASTNode = Union[Root, Moves, NodeList, Branch, Repeat, SymbolicNode, Variable, Node]
ASTFunctor = Callable[[ASTNode], Optional[ASTNode]]
ASTTerminal = Callable[[ASTNode, Optional[bool]], None]

SymbolTable = List[ASTNode]

#### Functions required to construct ASTs

#def compress_freeman(node: FreemanNode):
#    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False

def find_patterns_offset(node: Union[PatternNode, Segment], offset):
    """
    Find repeating node patterns of any size at the given start index,
    including alternating patterns, given it compresses the code
    """
    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False

    path = node.patterns
    length_pattern_max = (len(path) - offset + 1) // 2  # At least two occurrences are needed
    for length_pattern in range(1, length_pattern_max + 1):
        noffset = offset + length_pattern
        pattern = path[offset:noffset]
        for reverse in [False, True]:
            count = 1
            i = noffset
            while i < len(path):
                if i + length_pattern > len(path):
                    break

                match = True
                for j in range(length_pattern):
                    if reverse and count % 2 == 1:
                        if path[offset + j] != path[i + length_pattern - 1 - j]:
                            match = False
                            break
                    elif path[offset + j] != path[i + j]:
                        match = False
                        break

                if match:
                    count += 1
                    i += length_pattern
                else:
                    break

            if count > 1:
                len_original = sum(len(node) for node in pattern) * count
                len_compressed = len(RepeatNode(node=CompressedNode(pattern, set()), count=-count if reverse else count))
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse

def simplify_repetitions(node: PatternNode):
    simplified = []
    path = node.patterns
    i = 0
    while i < len(path):
        pattern, count, reverse = find_patterns_offset(path[i:], 0)
        if pattern is None or count <= 1:
            # No repeating pattern found, add the current move and continue
            simplified.append(path[i])
            i += 1
        else:
            if len(pattern) == 1:
                node = pattern[0]
            else:
                node = Moves(''.join(m.moves for m in pattern))
            if reverse:
                simplified.append(Repeat(node, -count))
            else:
                simplified.append(Repeat(node, count))
            i += len(pattern) * count

    if len(simplified) == 1:
        return simplified[0]
    else:
        result = []
        curr = simplified[0]
        for i in range(1, len(simplified)):
            next_node = simplified[i]
            if isinstance(curr, Moves) and isinstance(next_node, Moves):
                curr = Moves(curr.moves + next_node.moves)
            else:
                result.append(curr)
                curr = next_node
        result.append(curr)

        # If after concatenation we're left with a single Moves object
        # that's identical to self, return self
        if len(result) == 1 and isinstance(result[0], Moves) and result[0].moves == self.moves:
            return self

        return NodeList(result) if len(result) > 1 else result[0]

def len_param(param) -> int:
    if isinstance(param, ASTNode):
        return len(param)
    if isinstance(param, set):
        return LEN_COLOR
    if isinstance(param, tuple):
        return LEN_COORD
    if isinstance(param, list):
        return sum([len_param(p) for p in param])
    else:
        return 1

def rect_to_moves(height, width):
    moves='2' * (width - 1) + ''.join(
        '3' + ('0' if i % 2 else '2') * (width - 1)
        for i in range(1, height)
    )

    return Moves(moves)

def moves_to_rect(s: str):
    if not s:
        return None
    if len(s) <= 2:
        return None

    width = s.index('3')+1 if '3' in s else len(s) + 1
    height = s.count('3') + 1

    expected = '2' * (width - 1) + ''.join(
        '3' + ('0' if i % 2 else '2') * (width - 1)
        for i in range(1, height)
    )

    return (height, width) if s == expected and height >= 2 and width >= 2 else None

def expand(node) -> str:
    match node:
        case Moves(m):
            return m
        case Repeat(n, c) if isinstance(c, int):
            return c * expand(n)
        case UnionNode(nodes, _, _) | NodeList(nodes=nodes):
            expanded = []
            for n in nodes:
                e = expand(n)
                if e == '':
                    return ''
                expanded.append(e)
            return ''.join(expanded)
        case _:
            return ''

def extract_rects(node):
    ex = expand(node)
    if ex == '':
        return node
    else:
        res =  moves_to_rect(ex)
        if res:
            return Rect(res[0], res[1])
        else:
            return node



#### AST
def node_from_list(nodes: List[ASTNode]) -> Optional[ASTNode]:
    if not nodes:
        return None
    elif len(nodes) == 1:
        return nodes[0]
    return NodeList(nodes)
def encode_run_length(moves: Moves):
    """
    Run-Length Encoding (RLE) is used to compress Moves into Repeats of Moves.
    It's an elementary compression method used directly at the creation of AST representin branching chain codes.
    As it's very simple, it does not lead to risk of over optimisation
    """
    def create_node(move, count):
        if count >= 3:
            return Repeat(Moves(move), count)
        return Moves(move*count)

    seq = moves.moves
    if len(seq) <= 2:
        return Moves(seq)

    sequence = []
    move_prev = seq[0]
    move_current = seq[1]
    count = 2 if move_current == move_prev else 1
    non_repeat_moves = move_prev if move_current != move_prev else ""

    # Iterating over the 3-character sequences
    for move_next in seq[2:]:
        if move_next == move_current: # _, A, A case
            count += 1
        else: # *, A, B
            if count >= 3: # *, A, A, A, B case
                if non_repeat_moves: # Saving previous non-repeating moves
                    sequence.append(Moves(non_repeat_moves))
                    non_repeat_moves = ""
                sequence.append(create_node(move_current, count)) # storing the repetition
            else: # *, C, _, A, B case
                non_repeat_moves += move_current * count
            count = 1
        move_current = move_next

    # Last segment
    if count >= 3:
        if non_repeat_moves:
            sequence.append(Moves(non_repeat_moves))
        sequence.append(create_node(move_current, count))
    else:
        non_repeat_moves += move_current * count
        sequence.append(Moves(non_repeat_moves))

    return node_from_list(sequence)

def factorize_moves(node: ASTNode):
    if not isinstance(node, (Moves, NodeList)):
        return node
    if isinstance(node, NodeList):
        if len(node.nodes) == 1:
            return node.nodes[0]
        return node

    if len(node.moves) < 3:
        return node
    factorized = node.simplify_repetitions()
    return factorized
    #match factorized:
    #    case Repeat(n, c) if isinstance(n, Moves):
    #        return Repeat(encode_run_length(n), c)
    #    case Moves(m):
    #        return encode_run_length(factorized)
    #    case _:
    #        return factorized
    #        raise ValueError


def shift_moves(i: int, node):
    match node:
        case Moves(moves):
            return Moves(''.join([str((int(m)+i)%8) for m in moves]))
        case Root(s, c, node):
            return Root(s, c, shift_moves(i, node))
        case Repeat(node, c):
            return Repeat(shift_moves(i, node),c)
        case SymbolicNode(index, param, len_ref) if isinstance(param, ASTNode):
            return SymbolicNode(index, shift_moves(i, param), len_ref)
        case NodeList(nodes=nodes):
            return NodeList([shift_moves(i, n) for n in nodes])
        case Branch(sequences=nodes):
            return Branch([shift_moves(i, n) for n in nodes])
        case _:
            return node
def get_iterator(node_ls) -> List[ASTNode]:
    """
    This function tries to find iterators to encode as a single Repeat at the root of Branch or NodeList
    It use the fact that Branch or NodeList can't have a single Repeat as a child to assign a double meaning to it
    Yes, it's a bit hacky but it helps to increase the language expressiveness cost free by compressing
    enumerations
    """
    if not (1 < len(node_ls) <= 7):
        return node_ls

    prev = node_ls[0]
    curr = node_ls[1]

    increment = 1

    # To get an iterator, it should either iterate in a the direct or reverse sense
    if curr == shift_moves(-1, prev):
        increment=-1
    elif curr != shift_moves(1, prev):
        return node_ls

    for i in range(2, len(node_ls)):
        prev = node_ls[i-1]
        curr = node_ls[i]
        if curr != shift_moves(increment, prev):
            return node_ls

    # if we got this far, we got our iterator
    return [Repeat(node_ls[0], increment*len(node_ls))]

### Functions on ASTs
def get_depth(ast):
    depth=1
    match ast:
        case None:
            return 0
        case Root(_, _, n) | Repeat(n, _):
            depth += get_depth(n)
        case Branch(seq=ls) | NodeList(nodes=ls):
            depth += max([get_depth(n) for n in ls])
        case SymbolicNode(_, param=p, len_ref=l) if isinstance(p, ASTNode):
            depth += get_depth(p)
        case BiSymbolicNode(_, param1=p1, param2=p2, len_ref=l) if isinstance(p, ASTNode):
            depth += max([get_depth(p1), get_depth(p2)])
    return depth

def is_symbolic(ast):
    match ast:
        case SymbolicNode() | BiSymbolicNode():
            return True
        case Repeat(n, _) | Root(_, _, n):
            return is_symbolic(n)
        case Branch(sequences=nls) | NodeList(nodes=nls):
            return any([is_symbolic(n) for n in nls])
        case _:
            return False

def is_function(ast):
    match ast:
        case Variable():
            return True
        case SymbolicNode(_, param) if isinstance(param, ASTNode):
            return is_function(param)
        case Repeat(n, _) | Root(_, _, n):
            return is_function(n)
        case Branch(sequences=nls) | NodeList(nodes=nls):
            return any([is_function(n) for n in nls])
        case _:
            return False
def get_symbols(ast):
    match ast:
        case SymbolicNode(i, n, l):
            return get_symbols(n) + [i]
        case BiSymbolicNode(i, n1, n2, l):
            return get_symbols(n1) + get_symbols(n2) + [i]
        case Repeat(n, _) | Root(_, _, n):
            return get_symbols(n)
        case Branch(sequences=nls) | NodeList(nodes=nls):
            return [item for n in nls for item in get_symbols(n)]
        case _:
            return []

def ast_map(f: ASTFunctor, node: Optional[ASTNode]) -> Optional[ASTNode]:
    """
    Map a function from an single ASTNode to an single AST node to an entire AST.
    :param f: function to map
    :param node:
    """
    match node:
        case None:
            return None
        case Root(start=s, colors=c, node=n):
            nnode = ast_map(f, n)
            return f(Root(s, c, nnode))
        case Branch(sequences = seqs):
            nsequences = [nnode for seq in seqs if (nnode := ast_map(f, seq)) is not None]
            return f(Branch(nsequences))
        case NodeList(nodes = node_ls):
            try:
                nnodes = [nnode for n in node_ls if (nnode := ast_map(f, n)) is not None]
                return f(NodeList(nnodes))
            except NotImplementedError as e:
                print(f"Caught a NotImplementedError: {e}")
                print(f"Traceback: {print_trace(e)}")
                print(f"List of subnodes: ")
                for n in node_ls:
                    print(f"Node: {n}")
            return NotImplemented
        case UnionNode(codes, shadowed, background):
            ncodes = set([ncode for n in codes if (ncode := ast_map(f, n)) is not None])
            if shadowed:
                nshadowed = set([ncode for n in shadowed if (ncode := ast_map(f, n)) is not None])
            else:
                nshadowed = None
            nbackground = ast_map(f, background) if background else None
            return f(UnionNode(ncodes, nshadowed, nbackground))
        case Repeat(node=n, count=c):
            nnode = ast_map(f, n)
            if nnode is not None:
                return f(Repeat(nnode, c)) #if nnode is not None else None
            else:
                return None
        case SymbolicNode(index=i, param=p, len_ref=l) if isinstance(p, ASTNode):
            nnode = ast_map(f, p)
            return f(SymbolicNode(i, nnode, l))
        case BiSymbolicNode(index=i, param1=p1, param2=p2, len_ref=l) if isinstance(p1, ASTNode) \
        and isinstance(p2, ASTNode):
            nnode1 = ast_map(f, p1)
            nnode2 = ast_map(f, p2)
            return f(BiSymbolicNode(i, nnode1, nnode2, l))
        case BiSymbolicNode(index=i, param1=p1, param2=p2, len_ref=l) if isinstance(p1, ASTNode):
            nnode1 = ast_map(f, p1)
            return f(BiSymbolicNode(i, nnode1, p2, l))
        case BiSymbolicNode(index=i, param1=p1, param2=p2, len_ref=l) if isinstance(p2, ASTNode):
            nnode2 = ast_map(f, p2)
            return f(BiSymbolicNode(i, p1, nnode2, l))
        case _:
            return f(node)

### Helper functions to compress ASTs

def find_repeating_pattern(nodes: List[ASTNode], offset):
    """
    Find repeating node patterns of any size at the given start index,
    including alternating patterns, given it compresses the code
    """
    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False
    length_pattern_max = (len(nodes) - offset + 1) // 2  # At least two occurrences are needed
    for length_pattern in range(1, length_pattern_max + 1):
        noffset = offset + length_pattern
        pattern = nodes[offset:noffset]
        for reverse in [False, True]:
            count = 1
            i = noffset
            while i < len(nodes):
                if i + length_pattern > len(nodes):
                    break

                match = True
                for j in range(length_pattern):
                    if reverse and count % 2 == 1:
                        if nodes[offset + j] != nodes[i + length_pattern - 1 - j]:
                            match = False
                            break
                    elif nodes[offset + j] != nodes[i + j]:
                        match = False
                        break

                if match:
                    count += 1
                    i += length_pattern
                else:
                    break

            if count > 1:
                len_original = sum(len(node) for node in pattern) * count
                len_compressed = len(Repeat(node=NodeList(nodes=pattern), count=-count if reverse else count))
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse

def factorize_nodelist(ast_node):
    """
    Detect patterns inside a NodeList and factor them in Repeats.
    It takes a general AST Node parameter, as it makes it possible to construct a
    structural inducing version with ast_map
    """

    if not isinstance(ast_node, (NodeList, Branch)):
        return ast_node
    elif isinstance(ast_node, Branch):
        ast_node.check_iterators()
        return ast_node

    nodes = ast_node.nodes
    nnodes = []
    i = 0
    while i < len(nodes):
        pattern, count, reverse = find_repeating_pattern(nodes, i)
        # Use the square to make sure it's also ok for negative counts
        if count > 1:
            node = pattern[0] if len(pattern) == 1 else NodeList(nodes=pattern)
            if reverse:
                nnodes.append(Repeat(node, -count))
            else:
                nnodes.append(Repeat(node, count))
            i += len(pattern)*count
        else:
            nnodes.append(nodes[i])
            i += 1

    return NodeList(nodes=nnodes)

def functionalized(node: ASTNode) -> List[Tuple[ASTNode, Any]]:
    """
    Return a functionalized version of the node and a parameter.
    As the parameter count of Repeat is not a ASTNode, it's functionalized version
    is 0 to mark it needs to be replaced, -index once replaced
    """
    match node:
        case Branch(sequences=sequences):
            #max_sequence = max(sequences, key=len)
            max_sequences = sorted(sequences, key=len, reverse=True)[:2]
            max_sequence = max_sequences[0]
            nsequences1 = [seq if seq != max_sequence else Variable(0) for seq in sequences]

            nsequences2 = []
            for nnode in sequences:
                if nnode in max_sequences:
                    nsequences2.append(Variable(max_sequences.index(nnode)))
                else:
                    nsequences2.append(nnode)

            return [(Branch(nsequences1), max_sequence), (Branch(nsequences2), max_sequences)]
        case NodeList(nodes=nodes):
            #max_node = max(nodes, key=len)
            max_nodes = sorted(nodes, key=len, reverse=True)[:2]
            max_node = max_nodes[0]

            # Replacing only 1
            nnodes1 = [nnode if nnode != max_node else Variable(0)for nnode in nodes]

            # Or replacing 2 variables
            nnodes2 = []
            for nnode in nodes:
                if nnode in max_nodes:
                    nnodes2.append(Variable(max_nodes.index(nnode)))
                else:
                    nnodes2.append(nnode)
            return [(NodeList(nnodes1), max_node), (NodeList(nnodes2), max_nodes)]
        case Repeat(node=nnode, count=count):
            functions = []
            if not isinstance(nnode, Variable) and not isinstance(count, Variable):
                functions = [(Repeat(Variable(0), count), nnode), (Repeat(nnode, Variable(0)), count)]
            return functions
        case Root(start=s, colors=c, node=n):
            functions = []
            if True or (not isinstance(s, Variable) and not isinstance(n, Variable) and not isinstance(c, Variable)):
                functions = [
                    # Bar the Root to memorize position otherwise it defeats the purpose of "objectification"
                    (Root(Variable(0), c, n), s),
                    #(Root(s, c, Variable(0)), n),
                    (Root(Variable(0), c, Variable(1)), [s, n])
                ]
                if len(c) == 1:
                    functions.extend([
                        #(Root(s, Variable(0), n), c),
                        (Root(Variable(0), Variable(1), n), [s, c]),
                        #(Root(s, Variable(0), Variable(1)), [c, n])

                    ])
            return functions
        case Rect(height=h, width=w) if h==w:
            return [(Rect(Variable(0), Variable(0)), h)]
        case _:
            return []


def copy_ast(node: ASTNode) -> ASTNode:
    return ast_map(lambda node: node, node)
### Other helper functions
### Main functions
def update_node_i_by_j(node: ASTNode, i, j):
    match node:
        case SymbolicNode(index, param, l) if index == i:
            return SymbolicNode(j, param, l)
        case BiSymbolicNode(index, param1, param2, l) if index == i:
            return BiSymbolicNode(j, param1, param2, l)
        case _:
            return node

    #return node.copy()
def update_node(node, mapping):
    match node:
        case SymbolicNode(i, param, l):
            if DEBUG_ASTMAP:
                print("\n")
                print(f"Mapping node: {node}")
                print(f"to {SymbolicNode(mapping[i], param, l)}")
                print(f"With the mapping: {i} -> {mapping[i]}")
            #if isinstance(param, ASTNode):
            #    param = ast_map(lambda node: update_node(node, mapping), param)
            return SymbolicNode(mapping[i], param, l)

    return node.copy()
def construct_node(coordinates, is_valid: Callable[[Coord], bool], traversal="dfs"):
    seen = set([coordinates])
    def transitionsTower(coordinates):
        return available_transitions(is_valid, coordinates, MOVES[:4])
    def transitionsBishop(coordinates):
        return available_transitions(is_valid, coordinates, MOVES[4:])

    def add_move_left(move, node: Optional[ASTNode]):
        if not node:
            return Moves(move)
        else:
            return Moves(move) + node

    def dfs(coordinates) -> Optional[ASTNode]:
        branches = []
        branches_simplified = []
        for move, ncoordinates in transitionsTower(coordinates):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                node = dfs(ncoordinates)
                nnode = add_move_left(move, node)
                branches.append(nnode)
        for move, ncoordinates in transitionsBishop(coordinates):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                node = dfs(ncoordinates)
                nnode = add_move_left(move, node)
                branches.append(nnode)

        for node in branches:
            if isinstance(node, Moves):
                branches_simplified.append(encode_run_length(node))
            else:
                branches_simplified.append(node)

        if len(branches_simplified) > 1:
            return Branch(branches_simplified)
        elif len(branches_simplified) == 1:
            return branches_simplified[0]
        else:
            return None

    def bfs(coordinates):
        queue = []
        branches = []
        branches_simplified = []
        queue_tower = []
        queue_bishop = []
        for move, ncoordinates in transitionsTower(coordinates):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                queue_tower.append((move, ncoordinates))
        for move, ncoordinates in transitionsBishop(coordinates):
            if ncoordinates not in seen:
                seen.add(ncoordinates)
                queue_bishop.append((move, ncoordinates))
        for queue in (queue_tower, queue_bishop):
            for move, ncoordinates in queue:
                node = bfs(ncoordinates)
                nnode = add_move_left(move, node)
                branches.append(nnode)

        for node in branches:
            if isinstance(node, Moves):
                # Run-Lenght Encoding
                branches_simplified.append(encode_run_length(node))
            else:
                branches_simplified.append(node)

        if len(branches_simplified) > 1:
            return Branch(branches_simplified)
        elif len(branches_simplified) == 1:
            return branches_simplified[0]
        else:
            return None

    node = bfs(coordinates) if traversal=="bfs" else dfs(coordinates)
    return ast_map(extract_rects, ast_map(factorize_nodelist, node))

def symbolize_next(ast_ls: List[ASTNode], refs: SymbolTable, lattice_count=1) -> Tuple[List[ASTNode], bool]:
    # co_symbolize a list of ast
    pattern_encountered = {}

    def register_node(pattern, param=None):
        # pattern = (count, value, value_param, bi)
        bi = isinstance(param, list)
        if pattern in pattern_encountered:
            pattern_encountered[pattern] = (pattern_encountered[pattern][0] + 1, \
                pattern_encountered[pattern][1], pattern_encountered[pattern][2] + len_param(param), bi)
        else:
            pattern_encountered[pattern] = (1, len(pattern), len_param(param), bi)
    def discover_node(node):
        """
        Add a node to the dictionary of node, and its functionalized variants
        """
        if not isinstance(node, (SymbolicNode, Variable, Root, UnionNode, BiSymbolicNode)): #the not Variable is probably not useful
            register_node(node)
        funs = functionalized(node)

        for fun, parameter in funs:
            register_node(fun, parameter)

    def add_symbol(symb):
        index = len(refs) # we are adding a new symbol at the end

        def precize_symb(node):
            return node#update_symbol(node, 0)#index)#, -1) #Var(-1) ande Repeat(_, 0) marks unupdated node

        symb_precized = ast_map(precize_symb, symb)
        refs.append(symb_precized)
        return symb_precized

    # For each ast and templates already in the symbol table, discover every node
    for node in ast_ls + list(refs):
        if node is not None:
            discover_node(node)
            for n in node.breadth_iter():
                discover_node(n)

    # 6: cost of a symbolic node
    def bit_gained(value, count, bi=False):
        if not bi:
            value_symb = len(SymbolicNode(-1, None, 0))
        else:
            value_symb = len(BiSymbolicNode(-1, None, None, 0))
        compression = count*value - (count-1)*value_symb - value
        return compression#original - compressed

    compressable =[(node, bit_gained(value, count)) \
    for node, (count, value, value_param, bi) in pattern_encountered.items() if bit_gained(value, count, bi) > 0 \
   and count > lattice_count and node not in refs]

    if not compressable:
        return ast_ls, False

    max_symb, _ = max(compressable, key=lambda x: x[1])
    symb_precized = add_symbol(max_symb)


    def replace_by_symbol(node, symb):
        index = len(refs) - 1

        # If the symbol is a constant, and equal to the current node repl
        # Replace the current node by the symbol
        if node == max_symb:
            return SymbolicNode(index, None, len(max_symb))

        # Else, propragate the symbolic node
        match node:
            case Branch(sequences=sequences):
                node = Branch([replace_by_symbol(n, max_symb) for n in sequences])
            case NodeList(nodes=nodes):
                node = NodeList([replace_by_symbol(n, max_symb) for n in nodes])
            case Repeat(node=n, count=c):
                node = Repeat(replace_by_symbol(n, max_symb), c)
            case Root(start=s, colors=c, node=n):
                node = Root(s, c, replace_by_symbol(n, max_symb))


        funs = functionalized(node)
        # And test for functions
        for fun, parameter in funs:
            if fun == max_symb:
                if isinstance(parameter, list):
                    return BiSymbolicNode(index, parameter[0], parameter[1], len(max_symb))
                return SymbolicNode(index, parameter, len(max_symb))

        return node

    def replace_symb(node):
        return replace_by_symbol(node, symb_precized)

    for ref in refs:
        match ref:
            case Branch(sequences=seqs):
                ref.sequences = [replace_symb(n) for n in seqs]
            case NodeList(nodes=node_ls):
                ref.nodes = [replace_symb(n) for n in node_ls]
            case Repeat(node=n, count=c):
                ref.node = replace_symb(n)
            case Root(start=s, colors=c, node=n):
                ref.node = replace_symb(n)

    # Note you can only replace one at a time because a same node can be part of several 1-form
    return [replace_symb(node) for node in ast_ls], True

@handle_elements
def symbolize(ast_ls: List[ASTNode], refs: SymbolTable, lattice_count=1) -> List[ASTNode]:
    """Symbolize ast_ls as much as possible"""
    ast_ls, changed = symbolize_next(ast_ls, refs, lattice_count=1)
    while changed:
        ast_ls, changed = symbolize_next(ast_ls, refs, lattice_count=1)
    return ast_ls

def resolve_symbolic(node: ASTNode, refs: SymbolTable):
    def replace_parameter(node, param, index):
        """Replace a variable by the right parameter"""
        match node:
            case Variable(index=i) if i==index:
                if DEBUG_RESOLVE:
                    print(f"Replacing variable {node} by {param}")
                return param
            case Rect(height=Variable(index=i1), width=Variable(index=i2)) if i1==index and i2==index:
                return Rect(height=param, width=param)
            case Repeat(node=n, count=Variable(index=i)) if i == index:
                if DEBUG_RESOLVE:
                    print(f"Replacing repeat's count {node.count} by {param}")
                return Repeat(n, param)
            case Root(start=Variable(index=i), colors=c, node=n) if i == index:
                if DEBUG_RESOLVE:
                    print(f"Replacing roots's start {node.start} by {param}")
                return Root(param, c, n)
            case Root(start=s, colors=Variable(index=i), node=n) if i == index:
                if DEBUG_RESOLVE:
                    print(f"Replacing roots's color {node.colors} by {param}")
                return Root(s, param, n)
        return node

    def resolve_symbolic_node(node: ASTNode):
        """Replace a SymbolicNode by the ASTNode it represents"""
        match node:
            case BiSymbolicNode(index, param1, param2, l):
                param1 = ast_map(resolve_symbolic_node, param1) if isinstance(param1, ASTNode) else param1
                param2 = ast_map(resolve_symbolic_node, param2) if isinstance(param2, ASTNode) else param2
                template = refs[index]

                replace_param1 = lambda node: replace_parameter(node, param1, 0)#index)
                replace_param2 = lambda node: replace_parameter(node, param2, 1)#index)
                nnode = ast_map(replace_param1, template)
                nnode = ast_map(replace_param2, nnode)
                return nnode

            case SymbolicNode(index, param):
                if param is None:
                    return refs[index]
                else:
                    # First resolve param:
                    param = ast_map(resolve_symbolic_node, param) if isinstance(param, ASTNode) else param
                    template = refs[index]
                    replace_param = lambda node: replace_parameter(node, param, 0)#index)
                    nnode = ast_map(replace_param, template)
                    return nnode
            case _:
                return node

    return ast_map(resolve_symbolic_node, node)

@handle_elements
def unsymbolize(ast_ls: List[ASTNode], refs: SymbolTable):
    """
    Unsymbolize an AST: eliminates all Symbolic Nodes.

    1. First eliminates Symbolic Nodes from the symbol table refs: expand each symbol
    2. The Replace each SymbolicNode within each AST
    """
    # Copy the symbol table
    crefs = [copy_ast(ref) for ref in refs]

    # Define the resolve function on the copy
    def resolve(node):
        resolved = resolve_symbolic(node, crefs)
        #if is_symbolic(resolved):
        #            return resolve(resolved)  # Handle nested SymbolicNodes
        return resolved

    if DEBUG_UNSYMBOLIZE:
        print("\n")
        print("Original Symbol Table")
        for ref in crefs:
            print("Code: ", ref)
            print("Repr: ", ref.__repr__())
            print("---------------")
        print("\n")
    # Resolve the copy, crefs need to be updated at each iterations, otherwise further iterations
    # will used outdated refs which contains symbols
    for i, ref in enumerate(crefs):
        if DEBUG_UNSYMBOLIZE:
            print(f"Index: {i}, Ref: {ref}")
        crefs[i] = resolve(ref)
        if DEBUG_UNSYMBOLIZE:
            print(f"New ref: {crefs[i]}")
    #crefs = [resolve(ref) for ref in crefs]

    if DEBUG_UNSYMBOLIZE:
        print("\n")
        print("New Symbol Table")
        for ref in crefs:
            print("Code: ", ref)
            print("Repr: ", ref.__repr__())
            print("---------------")
        print("\n")

    # Resolved each ASTs using the resolved copy
    return [ast_map(resolve, ast) for ast in ast_ls]
def populate_old(grid, node, coordinates=(0,0), color=1, construction = None):
    # if the node is a rot with valid coord, then it takes the lead

    if DEBUG:
        print("\n")
        print("Call to populate() with following parameters:")
        print(f"Node: {node}")
        print(f"color: {color}")
        print(f"coordinates: {coordinates}")

    if node is None:
        return coordinates

    if isinstance(node, UnionNode):
        if DEBUG:
            print("Node dected as UnionNode")
        if node.background:
            populate_old(grid, node.background, coordinates, color, construction)
        for code in node.codes:
            populate_old(grid, code, coordinates, color, construction)
        return coordinates

    if isinstance(node, Root):
        if DEBUG:
            print("Node dected as Root")
        if not isinstance(node.start, Variable):
            if DEBUG:
                print(f"Start set to {node.start}")
            coordinates = node.start
        if len(node.colors)==1:
            if DEBUG:
                print(f"Color set to {node.colors}")
            color = list(node.colors)[0]
        if DEBUG:
            print(f"New node is {node.node}")

        node = node.node

    col, row = coordinates
    ncoordinates = coordinates
    grid[row][col] = color
    if isinstance(node, Moves):
        if DEBUG:
            print("Node dected as Moves")
            print(f"Coloring with color {color} the path {node.moves}")
        for col, row in node.iter_path(coordinates):
          grid[row][col] = color
          ncoordinates = (col, row)
    if isinstance(node, (Repeat, NodeList)):
        if DEBUG:
            print("Node dected as Repeat or NodeList")
        for nnode in node:
            ncoordinates = populate_old(grid, nnode, ncoordinates, color, construction)
    if isinstance(node, Branch):
        if DEBUG:
            print("Node dected as Branch")
        for nnode in node:
            populate_old(grid,nnode, ncoordinates, color, construction)

    if construction is not None:
        construction.append(copy(grid))
    return ncoordinates
def factor_by_refs(node: ASTNode, refs: SymbolTable):
    """A non-factorize node of an AST in an lattice could possibly match a symbol learned from another lattice.
    Thus it's necessary to check if each node doen't match a an already exiting symbol
    That was added in the symbol table by another lattice.
    """

    # Here the order is very important, for every symbol,
    # the functionalized version of the node as to be first tested against the symbol
    # before goin to the next symbol.
    # Otherwise it won't match the first symbolization process'

    def factor_node_by_ref(node, ref_index):
        if node == refs[ref_index]:
            return SymbolicNode(ref_index, None, len(refs[ref_index]))
        funs = functionalized(node)
        for fun, parameter in funs:
            if fun == refs[ref_index]:
                return SymbolicNode(ref_index, parameter, len(refs[ref_index]))
        return node

    def factor_node(node):
        # If the symbol is a constant, and equal to the current node repl
        # Replace the current node by the symbol
        for i in range(len(refs)):
            nnode = factor_node_by_ref(node, i)
            if isinstance(nnode, SymbolicNode):
                return nnode
        return node

    return ast_map(factor_node, node)
def map_refs(refs: SymbolTable, refs_common: SymbolTable) -> Tuple[List[int], SymbolTable]:
    mapping_ref = {}
    mapping = []

    # Checking first non symbolic
    while len(mapping_ref) < len(refs):
        for i, ref in enumerate(refs):
            if ref in mapping_ref:
                break
            # Treat the node if all symbols it's referencing have been treated'
            subsymbols = set(get_symbols(ref))
            if all([symb in mapping_ref for symb in subsymbols]):
                ref_updated = ref
                # First update all the sub symbolic node referencing the previously treated node
                for ss in subsymbols:
                    ref_updated = ast_map(lambda node: update_node_i_by_j(node, ss, mapping_ref[ss]), ref_updated)
                # The check if it already exist in the common symbol table
                # If yes then reference the new index
                if ref_updated in refs_common:
                    mapping_ref[i] = refs_common.index(ref_updated)
                # Else update it's index and add it to the common table
                else:
                    mapping_ref[i] = len(refs_common)
                    ref_updated = ast_map(lambda node: update_symbol(node, mapping_ref[i]), ref_updated)
                    refs_common.append(ref_updated)
    mapping = sorted([(i,j) for i, j in mapping_ref.items()], key=lambda x: x[0])
    mapping = [x[1] for x in mapping]
    return mapping, refs_common
def fuse_refs(refs_ls: List[SymbolTable]) -> Tuple[SymbolTable, List[int]]:
    nrefs_ls = [[symbol.copy() for symbol in refs] for refs in refs_ls]
    def map_symbols(refs, refs_common):
        # First update the variable references
        # For each symbol either it's already there from another symbol table
        # Or it needs to be added to the common symbol tables
        # then the  new index is added to the mapping from the old table to the common table
        # And the variable references to the previous nodes needs to be uodated
        mapping = []
        mapping_ref = {}

        to_do = []

        for ref in refs:
            # Updating the reference with the current mapping
            if ref in refs_common:
                nindex = nrefs.index(ref)
            else:
                nindex = len(refs_common)
                update = lambda node: update_symbol(node, nindex)
                ref_updated = ast_map(update, ref)
                #ref_updated = ast_map(lambda node: update_node(node, mapping), ref)
                #refs_common.append(ref_updated)
            mapping_ref[ref] = nindex

        #
            #mapping.append(nindex)
        # First find the mapping then update the nodes
        return reversed(mapping)

    nrefs = []
    mappings = []
    for refs in nrefs_ls:
        # Map the new symbol table and fill with placeholders
        mapping, nrefs = map_refs(refs, nrefs)
        mappings.append(mapping)
    return nrefs, mappings
def update_asts(ast_ls: List[ASTNode], nrefs: SymbolTable, mapping: List[int]):

    if DEBUG_ASTMAP:
        nast_ls = []
        print(f"\n")
        for i, ast in enumerate(ast_ls):
            print(f"----------------------")
            print(f"Updating AST n°{i}: {ast}")
            nast = ast_map(lambda node: update_node(node, mapping), ast)
            print(f"New AST: {nast}")
            nast_ls.append(nast)

        return nast_ls

    # First update refs by refs
    ast_ls = [ast_map(lambda node: update_node(node, mapping), ast) for ast in ast_ls]
    # Then check if a ref added from another node factorize some code
    ast_ls = [factor_by_refs(ast, nrefs) for ast in ast_ls]
    return ast_ls

def construct_union(code: Optional[ASTNode], codes: List[ASTNode], unions: List[ASTNode], refs: SymbolTable, box: Box):
    # Hypothesess:
        # union elements of union are always already unsymbolized and have a background
        # code elements of codes are symbolized
        # a code elements cannot be both standalone and in a union already as the depths wouldn't match
        # Issue: codes can already be unions because unions can contain unions...

    for i, u in enumerate(unions):
        if not u.background:
            raise ValueError(f"Union {u} sent without background")

    # Unsymbolized the symbolized code and background
    unsymbolized = unsymbolize([scode for scode in codes], refs)
    background = unsymbolize(code, refs) # type: ignore

    # Expel the unions from codes
    code_unions = set((ncode for ncode in unsymbolized if isinstance(ncode, UnionNode)))
    code_roots = set((ncode for ncode in unsymbolized if isinstance(ncode, Root)))


    # fuse code_unions with unions
    code_unions.update(unions)
    codes_dump = set()


    #codes = [c for c in codes if c is not None]
    #codes.sort(reverse=True, key=lambda x: len(x[1]))
    def remove_symbolized_code(input: Dict[Color,Set[Tuple['ASTNode', 'ASTNode']]]) -> Dict[Color,Set['ASTNode']]:
        codes={}
        for color, subcode_c in input.items():
            codes_color = set([unsymbolized for ncode, unsymbolized in subcode_c])
            codes[color] = codes_color
        return codes

    # First handle subunion:
        # cases
        # A shadowed shadowed element of one is an element of the other
        # -> Remove the background, dump all elements in codes

    # First retrieve all codes,
    # if a code appears more than once, then the unions that contains it intersects
    # they should be expanded. Thus only unions with novel codes should be kept
    subunions = set()

    # Count occurrences of each code
    # Also count each color to know which one to use for background normalization
    colors_count = defaultdict(set)
    codes_count = {}

    for u in code_unions:
        for ncode in u.codes:
            codes_count[ncode] = codes_count.get(ncode, 0) + 1

    # codes_dump can still contains union lol
    for u in code_unions:
        if any(codes_count[ncode] > 1 for ncode in u.codes) or (u.shadowed & codes_count.keys()):
            codes_dump.update(u.codes)
            codes_dump.update(u.shadowed)
        else:
            subunions.add(u)


    subcodes = defaultdict(set)

    for ncode in code_roots:
        c = next(iter(ncode.colors)) # type: ignore
        index = unsymbolized.index(ncode)
        subcodes[c].add((codes[index], ncode))
        if is_symbolic(ncode):
            print(f'From code_roots, Error: code symbolic: {ncode}')
        colors_count[c].add((codes[index], ncode))

    for ncode in codes_dump:
        c = next(iter(ncode.colors)) # type: ignore
        subcodes[c].add((factor_by_refs(ncode, refs), ncode))
        if is_symbolic(ncode):
            print(f'From codes_dump: Error: code symbolic: {ncode}')
        colors_count[c].add((factor_by_refs(ncode, refs), ncode))

    for u in code_unions:
        for ncode in u.codes:
            c = next(iter(ncode.colors)) # type: ignore
            colors_count[c].add((factor_by_refs(ncode, refs), ncode))
        c = next(iter(u.background.colors)) # type: ignore
        colors_count[c].add((factor_by_refs(u.background, refs), u.background))

    # background normalization
    # First, computing the description length for each colors
    # Then sorting them by description length
    # FInally, if the background is less than the heaviest description replace it by the background

    len_colors = []
    for color, codes_color in colors_count.items():
        len_color = 0
        len_color_symbolic = 0
        is_border = False
        for symbolized, unsymbolized in codes_color:
            # Issue: a backgrojnd can be symbolized here, strang
            _, points = decode(unsymbolized)
            if points and touches_border(points, box):
                is_border = True

            len_color += len(symbolized)
            len_color_symbolic += (0 if not isinstance(symbolized, (SymbolicNode, BiSymbolicNode)) else symbolized.len_ref)
        len_colors.append((color, len_color, len_color_symbolic, is_border))

    # / 5 is an evil arbitrary parameter necessary in case some node memorization is "too" efficient
    len_colors.sort(reverse=True, key=lambda x: (x[3], x[1] +  x[2]/5))

    nbackground = None
    nshadowed = None
    nsubunions = set()
    unions_to_remove = []

    if len_colors:
        color, length, len_ref, is_border = len_colors[0]
        if background and is_border and length + len_ref/5 > len(code):
            background.colors = {color}
            nbackground = background

            for union in subunions:
                if next(iter(union.background.colors)) == color or (union.codes&subcodes[color]) :
                    for code in union.codes:
                        c = next(iter(code.colors)) # type: ignore
                        subcodes[c].add((code, code))
                        unions_to_remove.append(union)
                    #for code in union.shadowed:
                    #    c = next(iter(code.colors)) # type: ignore
                    #    subcodes[c].add((code, code))
            nshadowed = set((unsymbolized for ncode, unsymbolized in subcodes[color]))
            del subcodes[color]

    for union in subunions:
        if not union in unions_to_remove:
            nsubunions.add(union)

    if subcodes:
        subcodes =  quotient_to_set(remove_symbolized_code(input=subcodes))
    else:
        subcodes = set()

    return UnionNode( nsubunions | subcodes, nshadowed, nbackground) # type: ignore

def hide_position(node: ASTNode) -> ASTNode:
    match node:
        case Root(s, c, n):
            return Root(Variable(-1), c, n)
        case SymbolicNode(i, p, l) if isinstance(p, tuple):
            return SymbolicNode(i, Variable(-1), l)

    return node

def hide_position_with_params(node):
    match node:
        case Root(s, c, n):
            return (Root(Variable(-1), c, n), s)
        case SymbolicNode(i, p, l) if isinstance(p, tuple):
            return (SymbolicNode(i, Variable(-1), l), p)

    return node

def get_hidden_pos(union_ls: List[UnionNode], node: ASTNode) -> List[Coord]:
    pos = []
    for union in union_ls:
        for nnode in union.codes:
            if hide_position(nnode) == node:
                match nnode:
                    case Root(s, c, n):
                        pos.append(s)
                    case SymbolicNode(i, p) if isinstance(p, tuple):
                        pos.append(p)
        if union.background and hide_position(union.background) == node:
            match union.background:
                case Root(s, c, n):
                    pos.append(s)
                case SymbolicNode(i, p) if isinstance(p, tuple):
                    pos.append(p)
    return pos

def get_invariants(unions: List[UnionNode]) -> Set[ASTNode]:
    # First get constant programs then tried relative/hidden position programs
    invariants = set()
    seen = set()
    codes = [[code for code in union.codes] for union in unions]
    codes_rel = [[hide_position(code) for code in union.codes.values] for union in unions]

    # Get the common programs
    codes = set.intersection(*map(set, codes))
    codes_rel = set.intersection(*map(set, codes_rel))

    # First tried constant / fixed position,
    # and mark the relative one as seen so it won't get double counted
    for code in codes:
        invariants.add(code)
        seen.add(hide_position(code))

    # Then add the relative programs if the constant version/excat match is not there
    for code in codes_rel:
        if code not in seen:
            invariants.add(code)

    return invariants

def shift_ast(shift: Coord, node: ASTNode):
    def displace(node):
        match node:
            case Root(s, c, n) if isinstance(s, tuple):
                return Root((s[0] + shift[0], s[1] + shift[1]), c, n)
            case SymbolicNode(i, p, l) if isinstance(p, tuple):
                return SymbolicNode(i, (p[0] + shift[0], p[1] + shift[1]), l)
        return node
    return ast_map(displace, node)

@optional
def decode(node: ASTNode, coordinates: Coord =(0,0), color: int = 1) -> Tuple[Coord, Points]:
    """
    Decode asts into grid without having a particular grid imposed.
    The AST should not be symbolic, it has to have its symbols resolved first.
    It shouldn't either be a parametrized symbol, i.e. of a function of Vars
    It enables to dynamically decide of the resulting grid proportions downstream.
    """
    if is_function(node):
        print(node)
        raise ValueError("A function cannot be decoded")
    if is_symbolic(node):
        raise ValueError("Trying to decode a node containing symbols. Please resolve it's symbols first")

    if node is None:
        return coordinates, set()

    # First search for top-level nodes: UnionNode and Root
    match node:
        case UnionNode(codes, _, background):
            # First getting the points of every code
            points = set([point for code in codes for point in decode(code)[1]])

            # Then getting the points of the backgroudn that don't conflicts with a points of the codes
            coords = points_to_coords(points)
            code_background = decode(background)[1] if background else set()

            for col, row, color in code_background:
                if not (col, row) in coords:
                    points.add((col, row, color))

            return coordinates, points
        case Root(start, colors, nnode):
            # Switching to the root node, and setting appropriate parameters
            if not isinstance(start, Variable):
                coordinates = start
            if not isinstance(colors, Variable) and len(colors) == 1:
                color = list(colors)[0]
            node = nnode

    # Then continue with traditional nodes
    col, row = coordinates
    ncoordinates = coordinates

    # Start point
    points = [(col, row, color)]

    if isinstance(node, Rect):
        node = rect_to_moves(node.height, node.width)

    # Searching and processing the non-symbolic traditional nodes
    match node:
        case Moves(moves):
            for col, row in node.iter_path(coordinates):
                points.append((col, row, color))
                ncoordinates = (col, row)
        case NodeList() | Repeat():
            for nnode in node:
                ncoordinates, npoints = decode(nnode, ncoordinates, color)
                points.extend(npoints)
        case Branch():
            for nnode in node:
                _, npoints = decode(nnode, ncoordinates, color)
                points.extend(npoints)

    return ncoordinates, set(points)

def node_to_grid(node: ASTNode) -> Grid:
    _, points = decode(node)
    return points_to_grid_colored(points)

def ast_distance(node1: Optional[ASTNode], node2: Optional[ASTNode], refs: SymbolTable) -> int:
    def list_edit_distance(list1, list2):
        m, n = len(list1), len(list2)

        # Initialize the dynamic programming matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill the first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the rest of the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # If the strings are the same, no operation needed
                if list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Calculate the cost of each operation
                    replace_cost = dp[i-1][j-1] + ast_distance(list1[i-1], list2[j-1], refs)
                    delete_cost = dp[i-1][j] + len(list1[i-1])
                    insert_cost = dp[i][j-1] + len(list2[j-1])

                    # Choose the operation with minimum cost
                    dp[i][j] = min(replace_cost, delete_cost, insert_cost)

        return dp[m][n]
    def set_edit_distance2(s1, s2):
        s1, s2 = set(s1), set(s2)

        # Strings present in both sets
        common = s1.intersection(s2)

        # Strings only in s1 or s2
        only_in_s1 = s1 - common
        only_in_s2 = s2 - common

        # Calculate the cost of adding/removing strings
        add_remove_cost = len(only_in_s1) + len(only_in_s2)

        # Calculate the minimum cost of editing strings
        edit_cost = 0
        if only_in_s1 and only_in_s2:
            # Create a matrix of edit distances between strings in only_in_s1 and only_in_s2
            edit_matrix = [[ast_distance(a, b, refs) for b in only_in_s2] for a in only_in_s1]

            while edit_matrix and any(edit_matrix):  # Check if matrix is not empty and has non-empty rows
                # Find the minimum edit distance
                min_dist = min(min(row) for row in edit_matrix if row)
                edit_cost += min_dist

                # Find the position of the minimum distance
                row_idx, col_idx = next((i, row.index(min_dist))
                                        for i, row in enumerate(edit_matrix)
                                        if row and min(row) == min_dist)
                col_idx = edit_matrix[row_idx].index(min_dist)

                # Remove the matched strings
                edit_matrix.pop(row_idx)
                for row in edit_matrix:
                    if row:  # Check if the row is not empty
                        row.pop(col_idx)

        return add_remove_cost + edit_cost

    def set_edit_distance1(s1, s2):
        s1, s2 = set(s1), set(s2)

        # Strings present in both sets
        common = s1.intersection(s2)

        # Strings only in s1 or s2
        only_in_s1 = s1 - common
        only_in_s2 = s2 - common

        # Calculate initial total cost
        total_cost = sum(len(ast) for ast in only_in_s1 if ast) + sum(len(ast) for ast in only_in_s2 if ast)

        if only_in_s1 and only_in_s2:
            # Create edit matrix with tuples: (edit_distance, s1_length, s2_length)
            edit_matrix = [
                [(ast_distance(a, b, refs), len(a) if a else 0, len(b) if b else 0)
                    for b in only_in_s2]
                for a in only_in_s1
            ]

            while edit_matrix and any(edit_matrix):  # Check if matrix is not empty and has non-empty rows
                # Find minimum edit distance
                min_dist, s1_len, s2_len = min(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[0]
                )

                # Find position of minimum distance
                row_idx, col_idx = next(
                    (i, row.index((min_dist, s1_len, s2_len)))
                    for i, row in enumerate(edit_matrix)
                    if (min_dist, s1_len, s2_len) in row
                )

                print(f'pairing: S1: {list(only_in_s1)[row_idx]} S2: {list(only_in_s2)[col_idx]}')
                print(f'dist and len: {edit_matrix[row_idx][col_idx]}')

                # Adjust total cost
                total_cost -= s1_len + s2_len
                total_cost += min_dist

                # Remove processed items
                edit_matrix.pop(row_idx)
                for row in edit_matrix:
                    if row:
                        row.pop(col_idx)

        return total_cost

    def set_edit_distance(s1, s2):
        s1, s2 = set(s1), set(s2)

        # Strings present in both sets
        common = s1.intersection(s2)

        # Strings only in s1 or s2
        only_in_s1 = list(s1 - common)  # Convert to list immediately
        only_in_s2 = list(s2 - common)  # Convert to list immediately

        # Calculate initial total cost
        total_cost = sum(len(ast) for ast in only_in_s1 if ast) + sum(len(ast) for ast in only_in_s2 if ast)

        if only_in_s1 and only_in_s2:
            # Create edit matrix with tuples: (edit_distance, s1_length, s2_length)
            edit_matrix = [
                [(ast_distance(a, b, refs), len(a) if a else 0, len(b) if b else 0)
                    for b in only_in_s2]
                for a in only_in_s1
            ]

            while edit_matrix and any(edit_matrix):  # Check if matrix is not empty and has non-empty rows
                # Find minimum edit distance
                min_dist, s1_len, s2_len = max(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[1] + x[2] - x[0]
                )

                # Check if edit would increase total cost
                if min_dist >= s1_len + s2_len:
                    break  # Stop if no beneficial edits remain

                # Find position of minimum distance
                row_idx, col_idx = next(
                    (i, row.index((min_dist, s1_len, s2_len)))
                    for i, row in enumerate(edit_matrix)
                    if (min_dist, s1_len, s2_len) in row
                )

                # Adjust total cost
                total_cost -= s1_len + s2_len
                total_cost += min_dist

                # Remove processed items
                edit_matrix.pop(row_idx)
                for row in edit_matrix:
                    if row:
                        row.pop(col_idx)

        return total_cost

    def node_list_distance(n, ls):
        min_dist, min_i = ast_distance(ls[0], n, refs), 0
        for i, nnode in enumerate(ls[1:]):
            if min_dist == 0:
                break
            dist = ast_distance(n, nnode, refs)
            if dist < min_dist:
                min_dist, min_i = dist, i
        return min_dist + sum([len(n) for n in (ls[:min_i] + ls[min_i+1:])])
    def param_distance(p1, p2):
        # if p1!=p2 and not ast_node, return greatest len
        match p1, p2:
            case (_, _) if isinstance(p1, ASTNode) and isinstance(p2, ASTNode):
                return ast_distance(p1, p2, refs)
            case (_, _) if p1 == p2:
                return 0
            case (_, _):
                return max(len_param(p1), len_param(p2))

    match node1, node2:
        case (None, None):
            return 0
        case (None, _):
            return len(node2)
        case (_, None):
            return len(node1)
        case (UnionNode(c1, s1, b1), UnionNode(c2, s2, b2)):
            #len_back = ast_distance(b1, b2, refs)
            len_codes = set_edit_distance(c1 | {b1}, c2 | {b2})
            return len_codes #len_back + len_codes
        case (UnionNode(c, s, b), _):
            len_b = len(b) if b else 0
            return min(node_list_distance(node2, c) + len_b, sum([len(n) for n in c]) + ast_distance(node2, b, refs))
        case (_, UnionNode(c, s, b)):
            return ast_distance(node2, node1, refs)
        case (SymbolicNode(i1, p1, l1), SymbolicNode(i2, p2, l2)):
            if i1==i2:
                return param_distance(p1, p2)
            else:
                return ast_distance(unsymbolize(node1, refs), unsymbolize(node2, refs), refs) #type :ignore
        case (SymbolicNode(i, p, l), _):
            udist = ast_distance(unsymbolize(node1, refs), node2, refs) # type: ignore
            sdist = 0
            if isinstance(p, ASTNode):
                sdist = LEN_NODE + LEN_INDEX + ast_distance(p, node2, refs)
            else:
                sdist = len(node2) + len_param(p)
            return min(udist, sdist)
        case (_, SymbolicNode(i, p, l)):
            return ast_distance(node2, node1, refs)
        case (BiSymbolicNode(i1, p11, p21, l1), BiSymbolicNode(i2, p12, p22, l2)):
            if i1 == i2:
                dist1 = param_distance(p11, p12)
                dist2 = param_distance(p21, p22)
                return dist1 + dist2
            return ast_distance(unsymbolize(node1, refs), unsymbolize(node2, refs), refs)
        case (BiSymbolicNode(i, p1, p2, l), _):
            dist1 = ast_distance(p1, node2, refs) if isinstance(p1, ASTNode) else len(node2) + len_param(p1)
            dist2 = ast_distance(p2, node2, refs) if isinstance(p2, ASTNode) else len(node2) + len_param(p2)
            sdist = LEN_INDEX + LEN_NODE
            if dist1 < dist2:
                sdist += dist1 + len(p2)
            else:
                sdist += dist2 + len(p1)
            return min(sdist, ast_distance(node2, unsymbolize(node1, refs), refs)) #type: ignore
        case (_, BiSymbolicNode(i, p1, p2, l)):
            return ast_distance(node2, node1, refs)
        case (Root(s1, c1, n1), Root(s2, c2, n2)):
            return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1]) + len(c1^c2)*LEN_COLOR + ast_distance(n1, n2, refs) # type: ignore
        case (Root(s, c, n), _):
            return LEN_COORD + LEN_COLOR*len(c) + ast_distance(n, node2, refs)
        case (_, Root(s, c, n)):
            return ast_distance(node2, node1, refs)
        case (Branch(nsequences=nls1), Branch(nsequences=nls2)):
            return set_edit_distance(nls1, nls2)
        case (Branch(nsequences=nls), _):
            return LEN_NODE + node_list_distance(node2, nls)
        case (_, Branch(nsequences=nls)):
            return ast_distance(node2, node1, refs)
        case (Rect(h1, w1), Rect(h2, w2)):
            return 0 if h1==h2 and w1==w2 else LEN_COORD
        case (Rect(h, w), _):
            #return min(ast_distance(node2, rect_to_moves(h, w), refs), LEN_COORD+len(node2))
            return ast_distance(node2, rect_to_moves(h, w), refs)
        case(_, Rect(h, w)):
            return ast_distance(node2, node1, refs)
        case (NodeList(nodes=nodes1), NodeList(nodes=nodes2)):
            return list_edit_distance(nodes1, nodes2)
        case (NodeList(nodes=nodes), Repeat(n, c)):
            dist_rep = LEN_COUNT + LEN_NODE + ast_distance(n, node1, refs)
            dist_nl = node_list_distance(node1, nodes)
            return min(dist_rep, dist_nl)
        case (Repeat(n, c), NodeList(nodes=nodes)):
            return ast_distance(node2, node1, refs)
        case (NodeList(nodes=nodes), _):
            return node_list_distance(node2, nodes)
        case (_, NodeList(nodes=nodes)):
            return ast_distance(node2, node1, refs)
        case (Repeat(n1, c1), Repeat(n2, c2)):
            dist = ast_distance(n1, n2, refs)
            dist_count = LEN_COUNT if c1 != c2 else 0
            return dist + dist_count
        case (Repeat(n,c), _):
            dist = LEN_NODE + LEN_COUNT + ast_distance(n, node2, refs)
            return dist
        case (_, Repeat(n,c)):
            return ast_distance(node2, node1, refs)
        case (Moves(m1), Moves(m2)):
            return distance_levenshtein(m1, m2)*LEN_MOVE
        case _:
            return 0 if node1 == node2 else len(node1) + len(node2)
