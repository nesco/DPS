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

GREEDY = False

from dataclasses import dataclass, asdict
from collections import deque
from enum import IntEnum

from math import ceil

from typing import Any, List, Union, Optional, Iterator, Callable, Set, Tuple, Dict, NewType, cast
from helpers import *

from time import sleep
from freeman import *

import sys
sys.setrecursionlimit(10000)

class BitLength(IntEnum):
    COORD = 10 # Base length for node type (3 bits) because <= 8 types
    COLOR = 4
    NODE = 3
    MOVE = 3 # Assuming 8-connectivity 3bits per move
    COUNT = 4   # counts of repeats should be an int between 2 and 9 or -2 and -8 (4 bits) ideally
    INDEX = 3 # should not be more than 8 so 3 bits
    INDEX_VARIABLE = 1 # Variable can be 0 or 1 so 1 bit
    RECT = 8

#### Abstract Syntax Tree Structure

#directions = {
#    '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
#    '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
#}


@dataclass(frozen=True)
class Node:
    """
    Abstract Node class used to stuff inherited methods.
    """

    def __len__(self) -> int:
           return BitLength.NODE  # Base length for node type (3 bits) because <= 8 types

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __add__(self, other) -> 'ASTNode':
        if isinstance(other, ConsecutiveNode):
            return ConsecutiveNode([self] + other.nodes)
        elif isinstance(other, (Variable, Root)):
            return NotImplemented
        elif issubclass(other.__class__, Node):
            return ConsecutiveNode([self, other])
        else:
            return NotImplemented

@dataclass(frozen=True)
class MovesNode(Node):
    """
    MovesNode store litteral non branching parts of a Freeman code Chain.
    It's a container for a string of octal digits, each indicating a direction in 8-cconectivity.
    The directions are defined by:
        directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
        }
    """

    moves: str = field(hash=True) #0, 1, 2, 3, 4, 5, 6, 7 (3 bits)

    def __len__(self) -> int:
        return super().__len__() + BitLength.MOVE * len(self.moves) # Assuming 8-coinnectivity 3bits per move
    def __str__(self) -> str:
        return self.moves
    def __add__(self, other): # right addition
        match other:
            case str():
                return MovesNode(self.moves + other)
            case MovesNode():
                return encode_run_length(MovesNode(self.moves + other.moves))
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

@dataclass(frozen=True)
class Rect(Node):
    height: Union[int, 'Variable'] = field(hash=True)
    width: Union[int, 'Variable'] = field(hash=True)

    def __len__(self) -> int:
        return super().__len__() + BitLength.RECT
    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"

@dataclass(frozen=True)
class Repeat(Node):
    node: 'ASTNode' = field(hash=True)
    count: Union[int, 'Variable'] = field(hash=True)  # should be an int between 2 and 17 (4 bits) ideally

    def __len__(self) -> int:
        return super().__len__() + len(self.node) + BitLength.COUNT
    def __str__(self):
            return f"({str(self.node)})*{{{self.count}}}"
    def __iter__(self) -> Iterator['ASTNode']:
        if self.node is not None:
            if isinstance(self.count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")
            # Handle negative values as looping
            elif self.count < 0:
                return (self.node if i%2==0 else reverse_node(self.node) for i in range(-self.count))
            else:
                return (self.node for _ in range(self.count))
        return iter(())
    def __add__(self, other):
        match (self, other):
            case (Repeat(node=n1, count=c1), Repeat(node=n2, count=c2)) if n1==n2:
                return Repeat(node=n1, count=c1+c2)
            case (Repeat(node=n1, count=c1), _) if n1 == other:
                return Repeat(node=n1, count=c1+1)
            case _:
                return super().__add__(other)

@dataclass(frozen=True)
class SequenceNode(Node):
    """Abstract Parent class of ConsecutiveNode and AlternativeNode"""
    nodes: list['ASTNode'] = field(default_factory=list, hash=True)

    def __len__(self) -> int:
        return sum(len(node) for node in self.nodes)
    def __iter__(self) -> Iterator['ASTNode']:
        return (node for node in self.nodes if node is not None)

@dataclass(frozen=True)
class ConsecutiveNode(SequenceNode):
    """
    ConsecutiveNode is a container representing a possibly Branching Freeman Code chain
    where subparts are encoded AST Nodes.
    As an abstract container designed to replace List() with an object having the right methods,
    it doesn't count for the length.
    """

    def __str__(self) -> str:
        return "".join(str(node) for node in self.nodes)
    def __repr__(self) -> str:
        return self.__class__.__name__ + "(nodes=" + self.nodes.__repr__() +")"
    def __hash__(self) -> int:
        return hash(self.__repr__())
    def __add__(self, other):
        if isinstance(other, ConsecutiveNode):
            nnodes = self.nodes + other.nodes
            return construct_consecutive_node(self.nodes + other.nodes)
        elif isinstance(other, str):
            return self.__add__(MovesNode(other))
        elif isinstance(other, MovesNode):
            if self.nodes and isinstance(self.nodes[-1], Node):
                nnodes = self.nodes[:-1] + [self.nodes[-1]+other]
            else:
                nnodes = self.nodes + [other]
            return construct_consecutive_node(nnodes)
        elif isinstance(other, Repeat):
            if self.nodes and isinstance(self.nodes[-1], Repeat) and \
            self.nodes[-1].node == other.node:
                nnodes = self.nodes[:-1] + [Repeat(other.node, self.nodes[-1].count + other.count)]
                return construct_consecutive_node(nnodes)
            else:
                return construct_consecutive_node(self.nodes + [other])
        elif isinstance(other, AlternativeNode):
            return construct_consecutive_node(self.nodes + [other])
        else:
            return NotImplemented
    def __iter__(self) -> Iterator['ASTNode']:
        return (node for node in self.nodes if node is not None)

@dataclass(frozen=True)
class AlternativeNode(SequenceNode):
    """
    Represents branching of the traversal of a connected component.
    As a single freeman code chain can't represent all connected components,
    it needs to have branching parts. This node encode the branching.

    As a branch is never a single node, it can be used to encode iterators through a Repeat.
    If a AlternativeNode contains a single repeat, then it will act like a positive or negative iterator
    """

    def __len__(self) -> int:
        return super().__len__() +  BitLength.NODE
    def __str__(self) -> str:
        if len(self.nodes) == 1 and isinstance(self.nodes[0], Repeat):
            return "[+" + str(self.nodes[0]) + "]"
        return "[" + ",".join(str(node) for node in self.nodes) +"]"
    def __add__(self, other):
        if isinstance(other, AlternativeNode):
            return AlternativeNode(self.nodes+other.nodes)
        else:
            return super().__add__(other)
    def __iter__(self) -> Iterator['ASTNode']:
        # It needs to handle the repeat used as an iterator case
        if len(self.nodes) == 1 and isinstance(self.nodes[0], Repeat):
            n = self.nodes[0].node
            count = self.nodes[0].count

            if isinstance(count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")

            i=1
            if isinstance(count, int) and count < 0:
                i=-1
                count = -count
            return (shift_moves(k*i, n) for k in range(count))
        return iter(self.nodes)
    def __repr__(self) -> str:
        return f"AlternativeNode(nodes={self.nodes!r})"
    def __hash__(self) -> int:
        return hash(self.__repr__())

@dataclass(frozen=True)
class Variable(Node):
    """
    This Node serves to reminds where SymbolicNodes parameters need to be pasted.
    """
    index: int # The hash of the replacement pattern to know it will replace this

    def __len__(self):
        return super().__len__() + BitLength.INDEX_VARIABLE
    def __str__(self):
        return f"Var({self.index})"
    def __repr__(self):
        return self.__class__.__name__
    def __eq__(self, other):
        return isinstance(other, Variable)

@dataclass(frozen=True)
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
        return  super().__len__() + BitLength.INDEX + len_param(self.param) # Not the super, it's a placeholder for something that already counts the +2
    def __str__(self) -> str:
        if self.param:
            return f" s_{self.index}({self.param}) "
        return f" s_{self.index} "
    def __hash__(self) -> int:
        return hash(self.__repr__())

@dataclass(frozen=True)
class BiSymbolicNode(Node):
    """
    Test to enable products, lots of things are product, particularly squares
    """
    index: int
    param1: Any
    param2: Any
    len_ref: int

    def __len__(self) -> int:
        return  super().__len__() + BitLength.INDEX + len_param(self.param1) + len_param(self.param2)

    def __str__(self) -> str:
        return f" s_{self.index}({self.param1}, {self.param2}) "
    def __hash__(self) -> int:
        return hash(self.__repr__())

    def copy(self):
        return BiSymbolicNode(self.index, self.param1.copy() if isinstance(self.param1, ASTNode) else self.param1 \
            , self.param2.copy() if isinstance(self.param2, ASTNode) else self.param2, self.len_ref)

@dataclass(frozen=True)
class Root(Node):
    """
    Node representing a path root. Note that the branches here can possibly lead to overlapping paths
    """
    start: Union[Coord, Variable]
    colors: Union[Colors, Variable]
    node: Optional['ASTNode']

    def __post_init(self):
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
        return BitLength.COORD + len(self.colors) * BitLength.COLOR + len_node
    def __add__(self, other):
        match other:
            case Variable():
                return NotImplemented
            case Root(start=(col, row), colors=c, child=n) if not isinstance(self.start, Variable):
                if self.start[0] == col and self.start[1] == row:
                    if self.colors != c or isinstance(self.colors, Variable) or isinstance(c, Variable):
                        raise NotImplementedError()
                    return Root((col, row), c, AlternativeNode([self.node, n]))
                return AlternativeNode([self, other])
            case Root(start=s, colors=c, child=n) if  (isinstance(self.start, Variable) or isinstance(s, Variable)):
                raise NotImplementedError()
        return Root(self.start, self.colors, AlternativeNode([self.node, other]))
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

@dataclass(frozen=True)
class RelativeRoot(Node):
    """
    Node representing a path root. Note that the branches here can possibly lead to overlapping paths
    """
    start: Union[Coord, Variable]
    colors: Union[Colors, Variable]
    node: Optional['ASTNode']

    def __post_init(self):
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
        return BitLength.COORD + len(self.colors) * BitLength.COLOR + len_node
    def __add__(self, other):
        match other:
            case Variable():
                return NotImplemented
            case Root(start=(col, row), colors=c, child=n) if not isinstance(self.start, Variable):
                if self.start[0] == col and self.start[1] == row:
                    if self.colors != c or isinstance(self.colors, Variable) or isinstance(c, Variable):
                        raise NotImplementedError()
                    return Root((col, row), c, AlternativeNode([self.node, n]))
                return AlternativeNode([self, other])
            case Root(start=s, colors=c, child=n) if  (isinstance(self.start, Variable) or isinstance(s, Variable)):
                raise NotImplementedError()
        return Root(self.start, self.colors, AlternativeNode([self.node, other]))
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
# Strategy diviser pour régner avec marginalisation + reconstruction


@dataclass()
class UnionNode():
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

#CompressedFreeman = Union[RepeatNode, CompressedNode]
############
## Types

ASTNode = Union[Root, MovesNode, ConsecutiveNode, AlternativeNode, Repeat, SymbolicNode, Variable, Node]
ASTFunctor = Callable[[ASTNode], Optional[ASTNode]]
ASTTerminal = Callable[[ASTNode, Optional[bool]], None]

SymbolTable = List[ASTNode]


### Basic functions

# string function

def alphabet_to_rolling_hashes_generator(alphabet: str) -> Callable:
    BASE = len(alphabet)
    MOD = 2477 # prime number to mod the hash values


    def string_to_hash_function(s: str) -> Callable:
        """
        Compute the rolling hashes for any string s
        with the correct alphabet size.

        :param s: the input string
        """
        hashes = [0] * (len(s) + 1)
        for i in range(len(s)):
            hashes[i+1] = (hashes[i] * BASE + alphabet.index(s[i])) % MOD

        def indices_to_hash(start: int, end: int) -> int:
            if end < start:
                raise ValueError(f"Invalid substring to hash: End: {end} < Start: {start}")
            return ( hashes[end] - hashes[start] * pow(BASE, end - start, MOD) ) % MOD

        return indices_to_hash

    return string_to_hash_function

string_to_hash_function = alphabet_to_rolling_hashes_generator("01234567")

def check_for_repeat2(s, dp, start, end, indice_to_hash, indice_to_hash_reversed) -> Optional[ASTNode]:
    substring_length = end - start
    n = len(s)

    # Check for all possible pattern length
    for length in range(1, substring_length//2 + 1):
        repeat_unit = s[start:start + length]
        times = substring_length // length
        remainder = substring_length % length
        if remainder != 0:
            continue

        # Check for positive repeat
        # Using hashes to compare
        pattern_hash = indice_to_hash(start, start + length)
        match = True
        reversed = False
        for t in range(1, times):
            offset = t * length
            current_hash = indice_to_hash(start + offset, start + offset + length)
            if current_hash != pattern_hash:
                match = False
                break

        if match:
            repeat_node = Repeat(repeat_unit, times)
            return repeat_node

       # One day do something for looping patterns here?

    # If nothing is found
    return None

def check_for_repeat1(s, dp, start, end, indice_to_hash, indice_to_hash_reversed) -> Optional[ASTNode]:
    substring_length = end - start
    n = len(s)
    alphabet = "01234567"

    # Check for all possible repeat unit lengths
    for length in range(1, substring_length // 2 + 1):
        repeat_unit = s[start:start + length]
        times = substring_length // length
        remainder = substring_length % length
        if remainder != 0:
            continue

        # Precompute hashes
        repeat_unit_hash = indice_to_hash(start, start + length)
        rev_start = n - (start + length)
        rev_end = n - start
        reverse_repeat_unit_hash = indice_to_hash_reversed(rev_start, rev_end)

        # Initialize match flags
        match_positive = True
        match_negative = True

        for t in range(times):
            offset = start + t * length
            current_hash = indice_to_hash(offset, offset + length)

            if match_positive:
                # Positive repeat comparison
                if current_hash != repeat_unit_hash:
                    match_positive = False  # No longer a positive repeat

            if match_negative:
                # Negative repeat comparison based on parity of t
                if t % 2 == 0:
                    expected_hash = repeat_unit_hash
                else:
                    expected_hash = reverse_repeat_unit_hash

                if current_hash != expected_hash:
                    match_negative = False  # No longer a negative repeat

            # Early exit if neither match is possible
            if not match_positive and not match_negative:
                break

        # Check if a positive repeat was found
        if match_positive:
            repeat_node = Repeat(MovesNode(repeat_unit), times)
            return repeat_node

        # Check if a negative repeat was found
        if match_negative and times % 2 == 0:
            repeat_count = - times
            repeat_node = Repeat(MovesNode(repeat_unit), repeat_count)
            return repeat_node

    # If nothing is found
    return None

Sequence = Union[list[ASTNode], str]
def check_for_repeat(sequence: Sequence) -> Optional[ASTNode]:
    """Check the biggest repeat at the current level"""
    n = len(sequence)

    # Check for all possible repeat unit lengths
    for length in range(1, n // 2 + 1):
        repeat_unit = sequence[:length]
        reverse_unit = reverse_sequence(repeat_unit)
        times = n // length
        remainder = n % length
        if remainder != 0:
            continue

        # Initialize match flags
        match_positive = True
        match_negative = True

        for t in range(times):
            offset = t * length

            def is_equal(s1, s2):
                return all(s1[i] == s2[i] for i in range(length))

            if match_positive:
                # Positive repeat comparison
                if not is_equal(sequence[offset:offset+length], repeat_unit):
                    match_positive = False  # No longer a positive repeat

            if match_negative:
                # Negative repeat comparison based on parity of t
                if t % 2 == 0:
                    if not is_equal(sequence[offset:offset+length], repeat_unit):
                        match_negative = False
                else:
                    if not is_equal(sequence[offset:offset+length], reverse_unit):
                        match_negative = False

            # Early exit if neither match is possible
            if not match_positive and not match_negative:
                break

        repeat_unit = encode_run_length(MovesNode(repeat_unit)) if isinstance(repeat_unit, str) \
        else construct_consecutive_node(repeat_unit)
        # Check if a positive repeat was found
        if match_positive:
            #repeat_node = Repeat(factorize(repeat_unit), times)
            repeat_node = Repeat(repeat_unit, times)
            return repeat_node

        # Check if a negative repeat was found
        if match_negative and times % 2 == 0:
            repeat_count = - times
            #repeat_node = Repeat(factorize(repeat_unit), repeat_count)
            repeat_node = Repeat(repeat_unit, repeat_count)
            return repeat_node

    # If nothing is found
    return None

def encode_string1(s: str) -> ASTNode:
    n: int = len(s)
    dp:list[Optional[ASTNode]] = [None] * (n + 1) # dp[i] stores the minimal encoding for s[0:i]
    dp[0] = ConsecutiveNode([])  # Empty encoding for empty string
    indice_to_hash = string_to_hash_function(s)
    indice_to_hash_reversed = string_to_hash_function(s[::-1])
    memo = {}

    # Check the optimal for s[0:i]
    # so the answer we are looking for is in dp[n]
    for i in range(1, n + 1):
        # Initialize with MovesNode
        current_best = dp[i - 1] + MovesNode(s[i-1])  #dp[i-1] is never None
        min_length = len(current_best)

        # Consider all possible splits
        # splits being dp[j] + s[j:i] (best s[0:j] +  a compression on s[j:i])
        for j in range(0, i):
            substring = s[j:i]
            key = (j, i)

            # if the best candidate for the substring has been memoïzed
            if key in memo:
                candidate = memo[key]
            else:
                candidate = None
                # check for the best candidate for repeats
                repeat_candidate = check_for_repeat(s, dp, j, i, indice_to_hash, indice_to_hash_reversed)
                if repeat_candidate:
                    candidate = repeat_candidate
                else:
                    # Encode as MovesNode
                    candidate = MovesNode(substring)
                candidate = dp[j] + candidate
                memo[key] = candidate

            if candidate and len(candidate) < min_length:
                current_best = candidate
                min_length = len(candidate)

        dp[i] = current_best

    return dp[n]

def check_for_repeat_within(s: str, start: int, end: int, dp) -> Optional[ASTNode]:
    substring = s[start:end]
    substring_length = end - start

    # Try all possible lengths of the repeat unit
    for length in range(1, substring_length // 2 + 1):
        unit = s[start:start + length]
        count = substring_length // length
        remainder = substring_length % length
        if remainder != 0:
            continue

        # Check if the substring is made up of repeats of the unit
        if all(s[start + i * length:start + (i + 1) * length] == unit for i in range(count)):
            # Factorize the unit recursively
            unit_node = dp(start, start + length)
            repeat_node = Repeat(unit_node, count)
            return repeat_node

    # No repeat found
    return None

def encode_string(s: str) -> ASTNode:
    n = len(s)
    memo = {}

    def dp(start: int, end: int) -> ASTNode:
        key = (start, end)
        if key in memo:
            return memo[key]

        # Base case: single character
        if end - start == 1:
            node = MovesNode(s[start:end])
            memo[key] = node
            return node

        # Initialize with MovesNode for the substring
        min_node = MovesNode(s[start:end])
        min_length = len(min_node)

        # Try to factorize the substring
        for mid in range(start + 1, end):
            left_node = dp(start, mid)
            right_node = dp(mid, end)
            combined_node = left_node + right_node
            combined_length = len(combined_node)
            if combined_length < min_length:
                min_node = combined_node
                min_length = combined_length

        # Check for repeats within the substring
        repeat_candidate = check_for_repeat_within(s, start, end, dp)
        if repeat_candidate and len(repeat_candidate) < min_length:
            min_node = repeat_candidate
            min_length = len(min_node)

        memo[key] = min_node
        return min_node

    return dp(0, n)

def combine(sequence_left: Sequence) -> ASTNode:
    pass

def factorize1(sequence: Sequence, memo = None) -> Optional[ASTNode]:
    def terminal_case(sequence:Union[list[ASTNode], str]) -> Optional[ASTNode]:
        if len(sequence) > 2:
            raise ValueError('Terminal case of factorization wrongly called')
        elif len(sequence) == 0:
            return None
        elif isinstance(sequence, str):
            return MovesNode(sequence)
        elif len(sequence) == 2:
            return ConsecutiveNode(sequence)
        else:
            return sequence[0]
    # sequence need to have a reverse, a len and an eq
    if len(sequence) < 3:
        return terminal_case(sequence)

    if memo is None:
        memo = {}

    seq_key = sequence if isinstance(sequence, str) else tuple(sequence)
    if seq_key in memo:
        return memo[seq_key]

    best_candidate = MovesNode(sequence) if isinstance(sequence, str) else ConsecutiveNode(sequence)
    min_length = len(best_candidate)

     # Try to factorize the entire sequence as a repeat
    candidate = check_for_repeat(sequence)
    if candidate:
        best_candidate = candidate
        min_length = len(candidate)
    # For every possible split, get the best left and right factorizations
    # and combine them by looking for repeating patterns
    for split in range(1, len(sequence) - 1):
        #left = check_for_repeat(sequence[:split])
        #right = check_for_repeat(sequence[split:])
        left = factorize(sequence[:split], memo)
        right = factorize(sequence[split:], memo)

        if left and right:
            candidate = left + right
        elif left:
            candidate = left
        elif right:
            candidate = right
        else:
            candidate = None

        if candidate and len(candidate) < min_length:
            best_candidate = candidate
            min_length = len(candidate)

    memo[seq_key] = best_candidate
    return best_candidate

def dynamic_factorize1(sequence: Sequence, memo = None) -> Optional[ASTNode]:
    def terminal_case(sequence:Union[list[ASTNode], str]) -> Optional[ASTNode]:
        if len(sequence) > 2:
            raise ValueError('Terminal case of factorization wrongly called')
        elif len(sequence) == 0:
            return None
        elif isinstance(sequence, str):
            return MovesNode(sequence)
        elif len(sequence) == 2:
            return ConsecutiveNode(sequence)
        else:
            return sequence[0]
    # sequence need to have a reverse, a len and an eq
    if len(sequence) < 3:
        return terminal_case(sequence)

    n = len(sequence)

    dp = [None] * (n + 1)
    dp_length = [float('inf')] * (n + 1)
    dp[0] = MovesNode('')  # Base case: empty sequence has zero length

    # Compute only the split that factors the most for every prefix of length n
    for i in range(1, n + 1):
        best_length = float('inf')
        best_candidate = None
        for j in range(i):
            subseq = sequence[j:i]
            # First, try to check for repeats
            repeat_result = check_for_repeat(subseq)
            if repeat_result:
                node = repeat_result
            else:
                node = MovesNode(subseq) if isinstance(subseq, str) else construct_consecutive_node(subseq)
            candidate = dp[j] + node
            length_candidate = len(candidate)
            if length_candidate < best_length:
                best_candidate = node
                best_length = length_candidate
        dp[i] = best_candidate

    return dp[n]





    return best_candidate

def dynamic_factorize(sequence: Sequence) -> Optional[ASTNode]:
    n = len(sequence)
    if n == 0:
        return None

    # Initialize the dp array to store the best node for each prefix
    dp = [None] * (n + 1)
    dp_length = [float('inf')] * (n + 1)
    dp[0] = MovesNode('') if isinstance(sequence, str) else ConsecutiveNode([])
    dp_length[0] = 0

    # Iterate over the sequence to build up the dp array
    for i in range(1, n + 1):
        for j in range(0, i):
            subseq = sequence[j:i]

            # try to find a repeat pattern
            repeat_result = check_for_repeat(subseq)
            if repeat_result:
                node = repeat_result
            else:
                # No repeat found; create a MovesNode or ConsecutiveNode
                if isinstance(subseq, str):
                    node = encode_run_length(MovesNode(subseq))
                else:
                    node = ConsecutiveNode(subseq)

             # Combine with the best solution up to position j
            if dp[j] is not None:
                combined_node = dp[j] + node
            else:
                combined_node = node  # Starting from the beginning

            total_length = len(combined_node)

            # Update dp[i] if a shorter encoding is found
            if total_length < dp_length[i]:
                dp[i] = combined_node
                dp_length[i] = total_length

    # The best encoding for the entire sequence is stored in dp[n]
    return dp[n]


def next_repeating_pattern(sequence: list[T], offset):
    """
    Find repeating pattern at offset, return (pattern, )
    including alternating patterns, given it compresses the code
    """
    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False
    length_pattern_max = (len(sequence) - offset + 1) // 2  # At least two occurrences are needed

    for length_pattern in range(1, length_pattern_max + 1):
        noffset = offset + length_pattern
        pattern = sequence[offset:noffset]
        for reverse in [False, True]:
            count = 1
            i = noffset
            while i < len(sequence):
                if i + length_pattern > len(sequence):
                    break

                match = True
                for j in range(length_pattern):
                    if reverse and count % 2 == 1:
                        if sequence[offset + j] != sequence[i + length_pattern - 1 - j]:
                            match = False
                            break
                    elif sequence[offset + j] != sequence[i + j]:
                        match = False
                        break

                if match:
                    count += 1
                    i += length_pattern
                else:
                    break

            if count > 1:
                len_original = sum(len(node) for node in pattern) * count
                len_compressed = len(Repeat(node=construct_consecutive_node(pattern), count=-count if reverse else count))
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse

def path_to_ast(path: Path) -> ASTNode:
    pass

def simplify_branch():
    pass

def simplify_repetitions(self):
    moves_ls = [MovesNode(k) for k in self.moves]
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
                node = MovesNode(''.join(m.moves for m in pattern))
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
            if isinstance(curr, MovesNode) and isinstance(next_node, MovesNode):
                curr = MovesNode(curr.moves + next_node.moves)
            else:
                result.append(curr)
                curr = next_node
        result.append(curr)

        # If after concatenation we're left with a single MovesNode object
        # that's identical to self, return self
        if len(result) == 1 and isinstance(result[0], MovesNode) and result[0].moves == self.moves:
            return self

        return construct_consecutive_node(result) if len(result) > 1 else result[0]

def construct_consecutive_node(nodes: list[ASTNode]) -> ConsecutiveNode:
    nodes_simplified = []

    current, nnodes = nodes[0], nodes[1:]
    while nnodes:
        next, nnodes = nnodes[0], nnodes[1:]

        ## Structural Pattern Matching
        match (current, next):
            case (Root(), _):
                raise NotImplementedError("Trying to initialise a node list with : " + ','.join(str(n) for n in nodes))
            case (MovesNode(), MovesNode()): # Concatening MovesNode
                current += next
            case (_, Repeat(node=n1, count=c1)) if current == n1:
                current = Repeat(n1, c1+1)
            case (ConsecutiveNode(), _): #Flattening ConsecutiveNode
                nodes_simplified.extend(current)
                current = next
            case (Repeat(node=n1, count=c1), Repeat(node=n2, count=c2)) if n1 == n2 and isinstance(c1, int) and isinstance(c2, int):  #Concatening Repeats
                current = Repeat(node=n1, count=c1+c2)
            # There can be nothing after a AlternativeNode, because it becomes implicitely a branch
            case (AlternativeNode(), _):
                while nnodes:
                    if isinstance(next, AlternativeNode):
                        current.nodes.extend(next.nodes)
                        next, nnodes = nnodes[0], nnodes[1:]
                    else:
                        current.nodes.append(ConsecutiveNode([next] + nnodes))
                        nnodes = None
                break  # Exit the main loop after processing the AlternativeNode (even though it should exit already)
            case _:
                nodes_simplified.append(current)
                current = next
    if isinstance(current, MovesNode):
        current = encode_run_length(current)
    if isinstance(current, ConsecutiveNode):
        nodes_simplified.extend(current)
    else:
        nodes_simplified.append(current)

    return ConsecutiveNode(nodes_simplified)

def breadth_iter(node: ASTNode) -> Iterator[ASTNode]:
    match node:
        case Repeat(nnode, count):
            yield nnode
            if isinstance(nnode, (AlternativeNode, Repeat, ConsecutiveNode)):
                yield from breadth_iter(nnode)
        case ConsecutiveNode(nodes=nodes):
            iterators = []
            for nnode in nodes:
                yield nnode
                if isinstance(nnode, (AlternativeNode, Repeat, ConsecutiveNode)):
                    iterators.append(breadth_iter(nnode))
            for it in iterators:
                yield from it
        case AlternativeNode(nodes=nodes):
            iterators = []
            for nnode in nodes:
                yield nnode
                if isinstance(nnode, (AlternativeNode, Repeat, ConsecutiveNode)):
                    iterators.append(breadth_iter(nnode))
            for it in iterators:
                yield from it
        case Root(start, colors, nnode) if nnode is not None:
            yield nnode
            if isinstance(nnode, (AlternativeNode, Repeat, ConsecutiveNode)):
                yield from breadth_iter(nnode)
        case _:
            return iter(())

def reverse_node(node: ASTNode) -> ASTNode:
    """Assume that AlternativeNode has already been run-lenght-encoded"""
    match node:
        case MovesNode(moves):
            return MovesNode(moves[::-1])
        case Repeat(nnode, count):
            return Repeat(reverse_node(nnode), count)
        case ConsecutiveNode(nodes=nodes):
            reverse_list = nodes[::-1]
            nnodes = [reverse_node(nnode) for nnode in reverse_list]
            return ConsecutiveNode(nnodes)
        case AlternativeNode(nodes=nodes):
            reverse_list = nodes[::-1]
            nnodes = [reverse_node(nnode) for nnode in reverse_list]
            return AlternativeNode(nnodes)
        case SymbolicNode(index, param, len_ref):
            if isinstance(param, ASTNode):
                param = reverse_node(param)
            return SymbolicNode(index, param, len_ref)
        case BiSymbolicNode(index, param1, param2, len_ref):
            if isinstance(param1, ASTNode):
                param1 = reverse_node(param1)
            if isinstance(param1, ASTNode):
                param2 = reverse_node(param2)
            return BiSymbolicNode(index, param1, param2, len_ref)
        case Root(start, colors, nnode) if nnode is not None:
                return Root(start, colors, reverse_node(nnode))
        case _:
            return node

def reverse_sequence(sequence: Sequence) -> Sequence:
    if isinstance(sequence, str):
        return sequence[::-1]
    if isinstance(sequence, list):
        return [reverse_node(node) for node in sequence[::-1]]
#### Functions required to construct ASTs

#def compress_freeman(node: FreemanNode):
#    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False


def len_param(param) -> int:
    if isinstance(param, ASTNode):
        return len(param)
    if isinstance(param, set):
        return BitLength.COLOR
    if isinstance(param, tuple):
        return BitLength.COORD
    if isinstance(param, list):
        return sum([len_param(p) for p in param])
    else:
        return 1

def rect_to_moves(height, width):
    moves='2' * (width - 1) + ''.join(
        '3' + ('0' if i % 2 else '2') * (width - 1)
        for i in range(1, height)
    )

    return MovesNode(moves)

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
        case MovesNode(m):
            return m
        case Repeat(n, c) if isinstance(c, int):
            return c * expand(n)
        case UnionNode(nodes, _, _) | ConsecutiveNode(nodes=nodes):
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
def node_from_list(nodes: list[ASTNode]) -> Optional[ASTNode]:
    if not nodes:
        return None
    elif len(nodes) == 1:
        return nodes[0]
    else:
        return ConsecutiveNode(nodes)

def branch_from_list(nodes: list[ASTNode]) -> Optional[ASTNode]:
    if not nodes:
        return None
    elif len(nodes) == 1 and not isinstance(nodes[0], Repeat):
        return nodes[0]
    else:
        return AlternativeNode(get_iterator(nodes))

def encode_run_length(moves: MovesNode):
    """
    Run-Length Encoding (RLE) is used to compress MovesNode into Repeats of MovesNode.
    It's an elementary compression method used directly at the creation of AST representin branching chain codes.
    As it's very simple, it does not lead to risk of over optimisation
    """
    def create_node(move, count):
        if count >= 3:
            return Repeat(MovesNode(move), count)
        return MovesNode(move*count)

    seq = moves.moves
    if len(seq) <= 2:
        return MovesNode(seq)

    sequence = []
    move_prev = seq[0]
    move_current = seq[1]
    count = 2 if move_current == move_prev else 1
    non_repeat_moves = move_prev if move_current != move_prev else ""

    # Iterating over the 3-character nodes
    for move_next in seq[2:]:
        if move_next == move_current: # _, A, A case
            count += 1
        else: # *, A, B
            if count >= 3: # *, A, A, A, B case
                if non_repeat_moves: # Saving previous non-repeating moves
                    sequence.append(MovesNode(non_repeat_moves))
                    non_repeat_moves = ""
                sequence.append(create_node(move_current, count)) # storing the repetition
            else: # *, C, _, A, B case
                non_repeat_moves += move_current * count
            count = 1
        move_current = move_next

    # Last segment
    if count >= 3:
        if non_repeat_moves:
            sequence.append(MovesNode(non_repeat_moves))
        sequence.append(create_node(move_current, count))
    else:
        non_repeat_moves += move_current * count
        sequence.append(MovesNode(non_repeat_moves))

    return node_from_list(sequence)

def factorize_moves(node: ASTNode):
    if not isinstance(node, (MovesNode, ConsecutiveNode)):
        return node
    if isinstance(node, ConsecutiveNode):
        if len(node.nodes) == 1:
            return node.nodes[0]
        return node

    if len(node.moves) < 3:
        return node
    factorized = node.simplify_repetitions()
    return factorized
    #match factorized:
    #    case Repeat(n, c) if isinstance(n, MovesNode):
    #        return Repeat(encode_run_length(n), c)
    #    case MovesNode(m):
    #        return encode_run_length(factorized)
    #    case _:
    #        return factorized
    #        raise ValueError

def shift_moves(i: int, node):
    match node:
        case MovesNode(moves):
            return MovesNode(''.join([str((int(m)+i)%8) for m in moves]))
        case Root(s, c, node):
            return Root(s, c, shift_moves(i, node))
        case Repeat(node, c):
            return Repeat(shift_moves(i, node),c)
        case SymbolicNode(index, param, len_ref) if isinstance(param, ASTNode):
            return SymbolicNode(index, shift_moves(i, param), len_ref)
        case ConsecutiveNode(nodes=nodes):
            return construct_consecutive_node([shift_moves(i, n) for n in nodes])
        case AlternativeNode(nodes=nodes):
            return AlternativeNode([shift_moves(i, n) for n in nodes])
        case _:
            return node
def get_iterator(node_ls) -> List[ASTNode]:
    """
    This function tries to find iterators to encode as a single Repeat at the root of AlternativeNode or ConsecutiveNode
    It use the fact that AlternativeNode or ConsecutiveNode can't have a single Repeat as a child to assign a double meaning to it
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
        case AlternativeNode(nodes=ls) | ConsecutiveNode(nodes=ls):
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
        case AlternativeNode(nodes=nls) | ConsecutiveNode(nodes=nls):
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
        case AlternativeNode(nodes=nls) | ConsecutiveNode(nodes=nls):
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
        case AlternativeNode(nodes=nls) | ConsecutiveNode(nodes=nls):
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
        case AlternativeNode(nodes = nodes):
            nnodes = [nnode for cnode in nodes if (nnode := ast_map(f, cnode)) is not None]
            return f(AlternativeNode(nnodes))
        case ConsecutiveNode(nodes = node_ls):
            try:
                nnodes = [nnode for n in node_ls if (nnode := ast_map(f, n)) is not None]
                return f(construct_consecutive_node(nnodes))
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
                len_compressed = len(Repeat(node=construct_consecutive_node(pattern), count=-count if reverse else count))
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse

def factorize_nodelist(ast_node):
    """
    Detect patterns inside a ConsecutiveNode and factor them in Repeats.
    It takes a general AST Node parameter, as it makes it possible to construct a
    structural inducing version with ast_map
    """

    if not isinstance(ast_node, (ConsecutiveNode, AlternativeNode)):
        return ast_node
    elif isinstance(ast_node, AlternativeNode):
        return AlternativeNode(get_iterator(ast_node.nodes))

    nodes = ast_node.nodes
    nnodes = []
    i = 0
    while i < len(nodes):
        pattern, count, reverse = find_repeating_pattern(nodes, i)
        # Use the square to make sure it's also ok for negative counts
        if count > 1:
            node = pattern[0] if len(pattern) == 1 else construct_consecutive_node(pattern)
            if reverse:
                nnodes.append(Repeat(node, -count))
            else:
                nnodes.append(Repeat(node, count))
            i += len(pattern)*count
        else:
            nnodes.append(nodes[i])
            i += 1

    return ConsecutiveNode(nodes=nnodes)

def functionalized(node: ASTNode) -> List[Tuple[ASTNode, Any]]:
    """
    Return a functionalized version of the node and a parameter.
    As the parameter count of Repeat is not a ASTNode, it's functionalized version
    is 0 to mark it needs to be replaced, -index once replaced
    """
    match node:
        case AlternativeNode(nodes=nodes):
            #max_sequence = max(nodes, key=len)
            max_nodes = sorted(nodes, key=len, reverse=True)[:2]
            max_node = max_nodes[0]
            nnodes1 = [nnode if nnode != max_node else Variable(0) for nnode in nodes]

            nnodes2 = []
            for nnode in nodes:
                if nnode in max_nodes:
                    nnodes2.append(Variable(max_nodes.index(nnode)))
                else:
                    nnodes2.append(nnode)

            return [(AlternativeNode(nnodes1), max_node), (AlternativeNode(nnodes2), max_nodes)]
        case ConsecutiveNode(nodes=nodes):
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
            return [(construct_consecutive_node(nnodes1), max_node), (construct_consecutive_node(nnodes2), max_nodes)]
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

    return node
def construct_node1(coordinates, is_valid: Callable[[Coord], bool], traversal: TraversalModes = TraversalModes.BFS):
    seen = set([coordinates])
    def transitionsTower(coordinates):
        return available_transitions(is_valid, coordinates, MOVES[:4])
    def transitionsBishop(coordinates):
        return available_transitions(is_valid, coordinates, MOVES[4:])

    def add_move_left(move, node: Optional[ASTNode]):
        if not node:
            return MovesNode(move)
        else:
            return MovesNode(move) + node

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
            if isinstance(node, MovesNode):
                branches_simplified.append(encode_run_length(node))
            else:
                branches_simplified.append(node)

        if len(branches_simplified) > 1:
            return AlternativeNode(branches_simplified)
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
            if isinstance(node, MovesNode):
                # Run-Lenght Encoding
                branches_simplified.append(encode_run_length(node))
            else:
                branches_simplified.append(node)

        if len(branches_simplified) > 1:
            return AlternativeNode(branches_simplified)
        elif len(branches_simplified) == 1:
            return branches_simplified[0]
        else:
            return None

    node = bfs(coordinates) if traversal==TraversalModes.BFS else dfs(coordinates)
    return ast_map(extract_rects, ast_map(factorize_nodelist, node))

def freeman_to_ast(freeman_node: FreemanNode) -> Optional[ASTNode]:
    branches = []
    nodelist = []

    for branch in freeman_node.branches:
        node = freeman_to_ast(branch)
        #if isinstance(node, AlternativeNode):
        #    node = AlternativeNode(get_iterator(node.nodes))#encode_run_length(node)
        branches.append(node)

    # And we compute the main path
    if len(freeman_node.path) > 0:
        if GREEDY:
            path = encode_run_length(MovesNode(''.join([str(move) for move in freeman_node.path])))
        else:
            path =  ''.join([str(move) for move in freeman_node.path])
            rect = extract_rects(MovesNode(path))
            if rect == MovesNode(path):
                #path = encode_string(path)
                path = dynamic_factorize(path)
            else:
                path = rect
        nodelist.append(path)

    branches = branch_from_list(branches)
    if branches is not None:
        nodelist.append(branches)

    astnode = node_from_list(nodelist)
    return astnode

def construct_node(freeman_node: FreemanNode) -> Optional[ASTNode]:
    # Recursive part: first we get the nodes from the different branches
    astnode = freeman_to_ast(freeman_node)
    if GREEDY:
        return ast_map(extract_rects, ast_map(factorize_nodelist, astnode))#ast_map(extract_rects, ast_map(factorize_nodelist, astnode))
    else:
        #return ast_map(factorize_nodelist, astnode)
        return astnode

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
            for n in breadth_iter(node):
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
            case AlternativeNode(nodes=nodes):
                node = AlternativeNode([replace_by_symbol(n, max_symb) for n in nodes])
            case ConsecutiveNode(nodes=nodes):
                node = ConsecutiveNode([replace_by_symbol(n, max_symb) for n in nodes])
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

    nrefs = list()
    for ref in refs:
        match ref:
            case AlternativeNode(nodes=nodes):
                nnodes = [replace_symb(n) for n in nodes]
                nrefs.append(AlternativeNode(nnodes))
            case ConsecutiveNode(nodes=nodes):
                nnodes = [replace_symb(n) for n in nodes]
                nrefs.append(AlternativeNode(nnodes))
            case Repeat(node=n, count=c):
                nrefs.append(Repeat(replace_symb(n), c))
            case Root(start=s, colors=c, node=n):
                nrefs.append(Root(s, c, replace_symb(n)))
            case _:
                nrefs.append(ref)

    refs = nrefs

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
    #refs = [copy_ast(ref) for ref in refs]
    crefs = refs

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
    nrefs_ls = [[symbol for symbol in refs] for refs in refs_ls]
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
            #background.colors = {color}
            nbackground = Root(background.start, {color}, background.node) #background

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
        case MovesNode(moves):
            for col, row in node.iter_path(coordinates):
                points.append((col, row, color))
                ncoordinates = (col, row)
        case ConsecutiveNode() | Repeat():
            for nnode in node:
                ncoordinates, npoints = decode(nnode, ncoordinates, color)
                points.extend(npoints)
        case AlternativeNode():
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

    def handle_none_cases(node1, node2):
        match node1, node2:
            case (None, None):
                return 0
            case (None, _):
                return len(node2)
            case (_, None):
                return len(node1)
            case _:
                raise ValueError(f'NoneCase wrongly called on {node1}, {node2}')

    match node1, node2:
        case (None, _) | (_, None):
            return handle_none_cases(node1, node2)
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
                sdist = BitLength.NODE + BitLength.INDEX + ast_distance(p, node2, refs)
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
            sdist = BitLength.INDEX + BitLength.NODE
            if dist1 < dist2:
                sdist += dist1 + len(p2)
            else:
                sdist += dist2 + len(p1)
            return min(sdist, ast_distance(node2, unsymbolize(node1, refs), refs)) #type: ignore
        case (_, BiSymbolicNode(i, p1, p2, l)):
            return ast_distance(node2, node1, refs)
        case (Root(s1, c1, n1), Root(s2, c2, n2)):
            return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1]) + len(c1^c2)*BitLength.COLOR + ast_distance(n1, n2, refs) # type: ignore
        case (Root(s, c, n), _):
            return BitLength.COORD + BitLength.COLOR*len(c) + ast_distance(n, node2, refs)
        case (_, Root(s, c, n)):
            return ast_distance(node2, node1, refs)
        case (AlternativeNode(nnodes=nls1), AlternativeNode(nnodes=nls2)):
            return set_edit_distance(nls1, nls2)
        case (AlternativeNode(nnodes=nls), _):
            return BitLength.NODE + node_list_distance(node2, nls)
        case (_, AlternativeNode(nnodes=nls)):
            return ast_distance(node2, node1, refs)
        case (Rect(h1, w1), Rect(h2, w2)):
            return 0 if h1==h2 and w1==w2 else BitLength.COORD
        case (Rect(h, w), _):
            #return min(ast_distance(node2, rect_to_moves(h, w), refs), LEN_COORD+len(node2))
            return ast_distance(node2, rect_to_moves(h, w), refs)
        case (_, Rect(h, w)):
            return ast_distance(node2, node1, refs)
        case (ConsecutiveNode(nodes=nodes1), ConsecutiveNode(nodes=nodes2)):
            return list_edit_distance(nodes1, nodes2)
        case (ConsecutiveNode(nodes=nodes), Repeat(n, c)):
            dist_rep = BitLength.COUNT + BitLength.NODE + ast_distance(n, node1, refs)
            dist_nl = node_list_distance(node2, nodes)
            return min(dist_rep, dist_nl)
        case (Repeat(n, c), ConsecutiveNode(nodes=nodes)):
            return ast_distance(node2, node1, refs)
        case (ConsecutiveNode(nodes=nodes), _):
            return node_list_distance(node2, nodes)
        case (_, ConsecutiveNode(nodes=nodes)):
            return ast_distance(node2, node1, refs)
        case (Repeat(n1, c1), Repeat(n2, c2)):
            dist = ast_distance(n1, n2, refs)
            dist_count = BitLength.COUNT if c1 != c2 else 0
            return dist + dist_count
        case (Repeat(n,c), _):
            dist = BitLength.NODE + BitLength.COUNT + ast_distance(n, node2, refs)
            return dist
        case (_, Repeat(n,c)):
            return ast_distance(node2, node1, refs)
        case (MovesNode(m1), MovesNode(m2)):
            return distance_levenshtein(m1, m2)*BitLength.MOVE
        case _:
            return 0 if node1 == node2 else len(node1) + len(node2)
