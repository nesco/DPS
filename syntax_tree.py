"""
Grid can be totally or partially encoded in a lossless fashion into Asbract Syntax Trees.
The lossless part is essential here because ARC Corpus sets the problem into the low data regime.
First grids are marginalized into connected components of N colors. Those connected components are "shapes".

Shapes are then represented the following way: extracting a possible branching pathes with a DFS on the connectivity graph,
then representing thoses branching pathes as a non-deterministic "program", where the output are string representing moves à la  freeman chain codes

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

from dataclasses import dataclass, fields
from collections import Counter
from enum import IntEnum


from typing import (
    Any,
    Union,
    Optional,
    Iterator,
    Callable,
    TypeVar,
    Generic,
)
from abc import ABC, abstractmethod
from helpers import *

from freeman import *
from grid import points_to_coords

from kolmogorov_tree import (
    KNode, BitLengthAware, KolmogorovTree,
    ProductNode, SumNode, RepeatNode, SymbolNode, RootNode, SymbolNode, PrimitiveNode,
    MoveValue, PaletteValue, IndexValue, VariableValue, CoordValue, BitLength,
    resolve_symbols, symbolize, children, breadth_iter, reverse_node, encode_run_length, is_function, contained_symbols
)


import sys

sys.setrecursionlimit(10000)

class BitLength(IntEnum):
    COORD = 10  # Base length for node type (3 bits) because <= 8 types
    COLOR = 4
    NODE = 3
    MOVE = 3  # Assuming 8-connectivity 3bits per move
    COUNT = 4  # counts of repeats should be an int between 2 and 9 or -2 and -8 (4 bits) ideally
    INDEX = 3  # should not be more than 8 so 3 bits
    INDEX_VARIABLE = 1  # Variable can be 0 or 1 so 1 bit
    RECT = 8

#### Abstract Syntax Tree Structure


# directions = {
#    '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
#    '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
# }
"""
@dataclass(frozen=True)
class BitLengthAware(ABC):
    ""
    This is a protocol for classes that 'know' their bit lengths.
    In the future it should be the length of the category they belong to.
    ""

    @abstractmethod
    def bit_length(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class Move(BitLengthAware):
    value: King

    def bit_length(self):
        return BitLength.MOVE

    def __str__(self) -> str:
        return str(self.value)


T = TypeVar("T", bound=BitLengthAware)


@dataclass(frozen=True)
class Node(Generic[T], ABC):
    ""
    Abstract Node class used to stuff inherited methods.
    ""

    def __len__(self) -> int:
        return BitLength.NODE  # Base length for node type (3 bits) because <= 8 types

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __add__(self, other) -> "Node":
        if isinstance(other, ConsecutiveNode):
            return ConsecutiveNode([self] + other.nodes)
        elif isinstance(other, (Variable, Root)):
            return NotImplemented
        elif issubclass(other.__class__, ASTNode):
            return ConsecutiveNode([self, other])
        else:
            return NotImplemented


@dataclass(frozen=True)
class Leaf(Node[T]):
    data: T

    def __len__(self) -> int:
        return super().__len__() + self.data.bit_length()

    def __str__(self) -> str:
        return str(data)

@dataclass(frozen=True)
class MovesNode(Node):
    ""
    MovesNode store litteral non branching parts of a Freeman code Chain.
    It's a container for a string of octal digits, each indicating a direction in 8-cconectivity.
    The directions are defined by:
        directions = {
            '0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1),
            '4': (1, -1), '5': (1, 1), '6': (-1, 1), '7': (-1, -1)
        }
    ""

    moves: str = field(hash=True)  # 0, 1, 2, 3, 4, 5, 6, 7 (3 bits)

    def __len__(self) -> int:
        return super().__len__() + BitLength.MOVE * len(
            self.moves
        )  # Assuming 8-coinnectivity 3bits per move

    def __str__(self) -> str:
        return self.moves

    def __add__(self, other):  # right addition
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
        ""
        Iterator over the path
        ""
        col, row = start
        for move in self.moves:
            col += DIRECTIONS[move][0]
            row += DIRECTIONS[move][1]
            yield (col, row)


@dataclass(frozen=True)
class Rect(Node):
    height: Union[int, "Variable"] = field(hash=True)
    width: Union[int, "Variable"] = field(hash=True)

    def __len__(self) -> int:
        return super().__len__() + BitLength.RECT

    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"


@dataclass(frozen=True)
class Repeat(Node):
    node: "ASTNode" = field(hash=True)
    count: Union[int, "Variable"] = field(
        hash=True
    )  # should be an int between 2 and 17 (4 bits) ideally

    def __len__(self) -> int:
        return super().__len__() + len(self.node) + BitLength.COUNT

    def __str__(self):
        return f"({str(self.node)})*{{{self.count}}}"

    def __iter__(self) -> Iterator["ASTNode"]:
        if self.node is not None:
            if isinstance(self.count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")
            # Handle negative values as looping
            elif self.count < 0:
                return (
                    self.node if i % 2 == 0 else reverse_node(self.node)
                    for i in range(-self.count)
                )
            else:
                return (self.node for _ in range(self.count))
        return iter(())

    def __add__(self, other):
        match (self, other):
            case (Repeat(node=n1, count=c1), Repeat(node=n2, count=c2)) if n1 == n2:
                return Repeat(node=n1, count=c1 + c2)
            case (Repeat(node=n1, count=c1), _) if n1 == other:
                return Repeat(node=n1, count=c1 + 1)
            case _:
                return super().__add__(other)

@dataclass(frozen=True)
class SequenceNode(Node):
    ""Abstract Parent class of ConsecutiveNode and AlternativeNode""

    nodes: list["ASTNode"] = field(default_factory=list, hash=True)

    def __len__(self) -> int:
        return sum(len(node) for node in self.nodes)

    def __iter__(self) -> Iterator["ASTNode"]:
        return (node for node in self.nodes if node is not None)


@dataclass(frozen=True)
class ConsecutiveNode(SequenceNode):
    ""
    ConsecutiveNode is a container representing a possibly Branching Freeman Code chain
    where subparts are encoded AST Nodes.
    As an abstract container designed to replace List() with an object having the right methods,
    it doesn't count for the length.
    ""

    def __str__(self) -> str:
        return "".join(str(node) for node in self.nodes)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(nodes=" + self.nodes.__repr__() + ")"

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
                nnodes = self.nodes[:-1] + [self.nodes[-1] + other]
            else:
                nnodes = self.nodes + [other]
            return construct_consecutive_node(nnodes)
        elif isinstance(other, Repeat):
            if (
                self.nodes
                and isinstance(self.nodes[-1], Repeat)
                and self.nodes[-1].node == other.node
            ):
                nnodes = self.nodes[:-1] + [
                    Repeat(other.node, self.nodes[-1].count + other.count)
                ]
                return construct_consecutive_node(nnodes)
            else:
                return construct_consecutive_node(self.nodes + [other])
        elif isinstance(other, AlternativeNode):
            return construct_consecutive_node(self.nodes + [other])
        else:
            return NotImplemented

    def __iter__(self) -> Iterator["ASTNode"]:
        return (node for node in self.nodes if node is not None)


@dataclass(frozen=True)
class AlternativeNode(SequenceNode):
    ""
    Represents branching of the traversal of a connected component.
    As a single freeman code chain can't represent all connected components,
    it needs to have branching parts. This node encode the branching.

    As a branch is never a single node, it can be used to encode iterators through a Repeat.
    If a AlternativeNode contains a single repeat, then it will act like a positive or negative iterator
    ""

    def __len__(self) -> int:
        return super().__len__() + BitLength.NODE

    def __str__(self) -> str:
        if len(self.nodes) == 1 and isinstance(self.nodes[0], Repeat):
            return "[+" + str(self.nodes[0]) + "]"
        return "[" + ",".join(str(node) for node in self.nodes) + "]"

    def __add__(self, other):
        if isinstance(other, AlternativeNode):
            return AlternativeNode(self.nodes + other.nodes)
        else:
            return super().__add__(other)

    def __iter__(self) -> Iterator["ASTNode"]:
        # It needs to handle the repeat used as an iterator case
        if len(self.nodes) == 1 and isinstance(self.nodes[0], Repeat):
            n = self.nodes[0].node
            count = self.nodes[0].count

            if isinstance(count, Variable):
                raise ValueError("An abstract Repeat cannot be expanded")

            i = 1
            if isinstance(count, int) and count < 0:
                i = -1
                count = -count
            return (shift_moves(k * i, n) for k in range(count))
        return iter(self.nodes)

    def __repr__(self) -> str:
        return f"AlternativeNode(nodes={self.nodes!r})"

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclass(frozen=True)
class Variable(Node):
    ""
    This Node serves to reminds where SymbolicNodes parameters need to be pasted.
    ""

    index: int  # The hash of the replacement pattern to know it will replace this

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
    ""
    To make the AST representations of branching code chains powerful,
    it needs to be able to compress efficiently code chains,
    with the possibility to memorize reoccuring patterns,
    be their constants or abstracted as functions of single parameters.
    The pattern needs to be stored independently in a list of patterns, index is the index of the pattern in this list
    ""

    index: int  # should not be more than 8 so 3 bits
    parameters: (
        tuple  #  Union[tuple, Variable] # for repeats and chained function calls
    )
    len_ref: int  # len of the ref

    def __len__(self) -> int:
        length = super().__len__() + BitLength.INDEX
        # if isinstance(self.parameters, Variable):
        #    length += len(self.parameters)
        # else:
        #    length += sum([len_param(param) for param in self.parameters])
        length += sum([len_param(param) for param in self.parameters])
        return length

    def __str__(self) -> str:
        if self.parameters:
            if isinstance(self.parameters, Variable):
                return f" s_{self.index}({self.parameters})"
            return f" s_{self.index}({', '.join(str(parameter) for parameter in self.parameters)})"
        return f" s_{self.index} "

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclass(frozen=True)
class Root(Node):
    ""
    Node representing a path root. Note that the branches here can possibly lead to overlapping paths
    ""

    start: Coord | Variable
    colors: Colors | Variable
    node: Optional["ASTNode"]

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
                print(
                    f"Trying to initialize {self} at nehative starting point {col}, {row}"
                )

    def __len__(self):
        #### Note: if start or colors are a Variable, the length shouldn't be counted
        # The length is already counted as the additional parameter
        # 10 bits for x and y going from 0 to 32 max on the grids + 4 bits for each color (10 choices)
        len_node = len(self.node) if self.node is not None else 0
        len_coord = (
            BitLength.COORD if not isinstance(self.start, Variable) else len(self.start)
        )
        len_colors = (
            BitLength.COLOR * len(self.colors)
            if not isinstance(self.colors, Variable)
            else len(self.colors)
        )

        return (
            BitLength.COORD + len_node + BitLength.COLOR * len(self.colors)
        )  # len_coord + len_colors + len_node

    def __add__(self, other):
        match other:
            case Variable():
                return NotImplemented
            case Root(start=(col, row), colors=c, child=n) if not isinstance(
                self.start, Variable
            ):
                if self.start[0] == col and self.start[1] == row:
                    if (
                        self.colors != c
                        or isinstance(self.colors, Variable)
                        or isinstance(c, Variable)
                    ):
                        raise NotImplementedError()
                    return Root((col, row), c, AlternativeNode([self.node, n]))
                return AlternativeNode([self, other])
            case Root(start=s, colors=c, child=n) if (
                isinstance(self.start, Variable) or isinstance(s, Variable)
            ):
                raise NotImplementedError()
        return Root(self.start, self.colors, AlternativeNode([self.node, other]))

    def __str__(self):
        return f"{self.colors}->{self.start}:" + str(self.node)

    # def __repr__(self):
    #    return f"{self.__class__.__name__}(start={self.start.__repr__()}, colors={str(self.colors)}, node={self.node.__repr__()})"
    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if not isinstance(other, Root):
            return False
        # if isinstance(self.start, Variable):
        #    return (self.colors==other.colors) and (self.node == other.node)
        return (
            (self.start == other.start)
            and (self.colors == other.colors)
            and (self.node == other.node)
        )

"""

@dataclass(frozen=True)
class Rect(KNode):
    height: 'int | VariableNode'  # Use VariableNode from kolmogorov_tree.py
    width: 'int | VariableNode'

    def bit_length(self) -> int:
        # 3 bits for node type + 8 bits for ARC-specific rectangle encoding
        height_len = BitLength.COUNT if isinstance(self.height, int) else self.height.bit_length()
        width_len = BitLength.COUNT if isinstance(self.width, int) else self.width.bit_length()
        return super().__len__() + height_len + width_len

    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"

# Strategy diviser pour régner avec marginalisation + reconstruction

@dataclass()
class UnionNode:
    """
    Represent a connected component by reconstructing it with the best set of single color programs.
    After marginalisation comes reconstruction, divide and conquer.
    """

    # background: dict['ASTNode', 'ASTNode']
    codes: set[KNode]
    shadowed: set[int] | None = None
    background: KNode | None = None

    def __len__(self):
        len_codes = 0
        if self.codes is None:
            return 0
        return sum([len(code) for code in self.codes])

    def __add__(self, other):
        raise NotImplementedError

    def __str__(self):
        msg = ""
        if self.background is not None:
            msg += f"{self.background } < "
        if self.codes is None:
            msg += "Ø"
        else:
            msg += "U".join([f"{{{code}}}" for code in self.codes])
        return msg

    def __repr__(self):
        codes_repr = ",".join(code.__repr__() for code in self.codes)
        return f"{self.__class__.__name__}(codes={codes_repr})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, UnionNode):
            return False
        return self.background == other.background and self.codes == other.codes

    def __hash__(self):
        return hash(self.__repr__())

"""
@dataclass()
class UnionNode:
    ""
    Represent a connected component by reconstructing it with the best set of single color programs.
    After marginalisation comes reconstruction, divide and conquer.
    ""

    # background: dict['ASTNode', 'ASTNode']
    codes: set["ASTNode"]
    shadowed: Optional[set[int]] = None
    background: Optional["ASTNode"] = None

    def __len__(self):
        len_codes = 0
        if self.codes is None:
            return 0
        return sum([len(code) for code in self.codes])

    def __add__(self, other):
        raise NotImplementedError

    def __str__(self):
        msg = ""
        if self.background is not None:
            msg += f"{self.background } < "
        if self.codes is None:
            msg += "Ø"
        else:
            msg += "U".join([f"{{{code}}}" for code in self.codes])
        return msg

    def __repr__(self):
        codes_repr = ",".join(code.__repr__() for code in self.codes)
        return f"{self.__class__.__name__}(codes={codes_repr})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, UnionNode):
            return False
        return self.background == other.background and self.codes == other.codes

    def __hash__(self):
        return hash(self.__repr__())


############## TEST : NEW FREEMAN STRUCTURE ###########

# CompressedFreeman = Union[RepeatNode, CompressedNode]
############
## Types


ASTNode = Union[
    Root,
    MovesNode,
    ConsecutiveNode,
    AlternativeNode,
    Repeat,
    SymbolicNode,
    Variable,
    Node,
]
ASTFunctor = Callable[[Node], Optional[Node]]
ASTTerminal = Callable[[Node, Optional[bool]], None]
"""

### Basic functions


def is_leaf_list(input) -> bool:
    if isinstance(input, ProductNode):
        node_list = input.children
    elif isinstance(input, list):
        node_list = input
    else:
        return False
    return all(isinstance(n, PrimitiveNode) for n in node_list)


Sequence = list[KNode] | str


def check_for_repeat(sequence: Sequence) -> KNode | None:
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
                if not is_equal(sequence[offset : offset + length], repeat_unit):
                    match_positive = False  # No longer a positive repeat

            if match_negative:
                # Negative repeat comparison based on parity of t
                if t % 2 == 0:
                    if not is_equal(sequence[offset : offset + length], repeat_unit):
                        match_negative = False
                else:
                    if not is_equal(sequence[offset : offset + length], reverse_unit):
                        match_negative = False

            # Early exit if neither match is possible
            if not match_positive and not match_negative:
                break

        repeat_unit = (
            encode_run_length(MovesNode(repeat_unit))
            if isinstance(repeat_unit, str)
            else construct_consecutive_node(repeat_unit)
        )
        # Check if a positive repeat was found
        if match_positive:
            # repeat_node = Repeat(factorize(repeat_unit), times)
            repeat_node = Repeat(repeat_unit, times)
            return repeat_node

        # Check if a negative repeat was found
        if match_negative and times % 2 == 0:
            repeat_count = -times
            # repeat_node = Repeat(factorize(repeat_unit), repeat_count)
            repeat_node = Repeat(repeat_unit, repeat_count)
            return repeat_node

    # If nothing is found
    return None


def check_for_repeat_within(s: str, start: int, end: int, dp) -> KNode | None:
    substring = s[start:end]
    substring_length = end - start

    # Try all possible lengths of the repeat unit
    for length in range(1, substring_length // 2 + 1):
        unit = s[start : start + length]
        count = substring_length // length
        remainder = substring_length % length
        if remainder != 0:
            continue

        # Check if the substring is made up of repeats of the unit
        if all(
            s[start + i * length : start + (i + 1) * length] == unit
            for i in range(count)
        ):
            # Factorize the unit recursively
            unit_node = dp(start, start + length)
            repeat_node = Repeat(unit_node, count)
            return repeat_node

    # No repeat found
    return None


def encode_string(s: str) -> KNode:
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


def dynamic_factorize(sequence: Sequence) -> KNode | None:
    n = len(sequence)
    if n == 0:
        return None

    # Initialize the dp array to store the best node for each prefix
    dp = [None] * (n + 1)
    dp_length = [float("inf")] * (n + 1)
    dp[0] = MovesNode("") if isinstance(sequence, str) else ConsecutiveNode([])
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
    length_pattern_max = (
        len(sequence) - offset + 1
    ) // 2  # At least two occurrences are needed

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
                len_compressed = len(
                    Repeat(
                        node=construct_consecutive_node(pattern),
                        count=-count if reverse else count,
                    )
                )
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse


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
                node = MovesNode("".join(m.moves for m in pattern))
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
        if (
            len(result) == 1
            and isinstance(result[0], MovesNode)
            and result[0].moves == self.moves
        ):
            return self

        return construct_consecutive_node(result) if len(result) > 1 else result[0]

"""
def construct_consecutive_node(nodes: list[KNode]) -> ConsecutiveNode:
    nodes_simplified = []

    current, nnodes = nodes[0], nodes[1:]
    while nnodes:
        next, nnodes = nnodes[0], nnodes[1:]

        ## Structural Pattern Matching
        match (current, next):
            case (Root(), _):
                raise NotImplementedError(
                    "Trying to initialise a node list with : "
                    + ",".join(str(n) for n in nodes)
                )
            case (MovesNode(), MovesNode()):  # Concatening MovesNode
                current += next
            case (_, Repeat(node=n1, count=c1)) if current == n1:
                current = Repeat(n1, c1 + 1)
            case (ConsecutiveNode(), _):  # Flattening ConsecutiveNode
                nodes_simplified.extend(current)
                current = next
            case (
                Repeat(node=n1, count=c1),
                Repeat(node=n2, count=c2),
            ) if n1 == n2 and isinstance(c1, int) and isinstance(
                c2, int
            ):  # Concatening Repeats
                current = Repeat(node=n1, count=c1 + c2)
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
"""

def reverse_sequence(sequence: Sequence) -> Sequence:
    if isinstance(sequence, str):
        return sequence[::-1]
    if isinstance(sequence, list):
        return [reverse_node(node) for node in sequence[::-1]]


#### Functions required to construct ASTs

# def compress_freeman(node: FreemanNode):
#    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False

"""
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
"""

# Refactored
def rect_to_moves(height, width) -> list[PrimitiveNode[MoveValue]]:
    moves = "2" * (width - 1) + "".join(
        "3" + ("0" if i % 2 else "2") * (width - 1) for i in range(1, height)
    )
    primitives = [PrimitiveNode(MoveValue(m)) for m in moves]
    return primitives

# To refactor
def moves_to_rect(s: str):
    if not s:
        return None
    if len(s) <= 2:
        return None

    width = s.index("3") + 1 if "3" in s else len(s) + 1
    height = s.count("3") + 1

    expected = "2" * (width - 1) + "".join(
        "3" + ("0" if i % 2 else "2") * (width - 1) for i in range(1, height)
    )

    return (height, width) if s == expected and height >= 2 and width >= 2 else None


# To refactor
def expand(node) -> str:
    match node:
        case ConsecutiveNode(nodes) if all(
            isinstance(n, Leaf) and isinstance(n.data, Move) for n in nodes
        ):
            return "".join([str(n.data.value) for n in nodes])
        case list if all(
            isinstance(n, Leaf) and isinstance(n.data, Move) for n in node
        ):
            return "".join([str(n.data.value) for n in node])
        case MovesNode(m):
            return m
        case Repeat(n, c) if isinstance(c, int):
            return c * expand(n)
        case UnionNode(nodes, _, _) | ConsecutiveNode(nodes=nodes):
            expanded = []
            for n in nodes:
                e = expand(n)
                if e == "":
                    return ""
                expanded.append(e)
            return "".join(expanded)
        case _:
            return ""


def extract_rects1(node):
    ex = expand(node)
    if ex == "":
        return node
    else:
        res = moves_to_rect(ex)
        if res:
            return Rect(res[0], res[1])
        else:
            return node

# To refactor
def extract_rects(node):
    ex = expand(node)
    if ex == "":
        return node
    else:
        res = moves_to_rect(ex)
        if res:
            return Rect(res[0], res[1])
        else:
            return node


#### AST
"""
def factorize_moves(node: Node):
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
    # match factorized:
    #    case Repeat(n, c) if isinstance(n, MovesNode):
    #        return Repeat(encode_run_length(n), c)
    #    case MovesNode(m):
    #        return encode_run_length(factorized)
    #    case _:
    #        return factorized
    #        raise ValueError
"""

### Helper functions to compress ASTs

"""
def find_repeating_pattern(nodes: list[Node], offset):
    ""
    Find repeating node patterns of any size at the given start index,
    including alternating patterns, given it compresses the code
    ""
    best_pattern, best_count, best_bit_gain, best_reverse = None, 0, 0, False
    length_pattern_max = (
        len(nodes) - offset + 1
    ) // 2  # At least two occurrences are needed
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
                len_compressed = len(
                    Repeat(
                        node=construct_consecutive_node(pattern),
                        count=-count if reverse else count,
                    )
                )
                bit_gain = len_original - len_compressed
                if best_bit_gain < bit_gain:
                    best_pattern = pattern
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count, best_reverse

"""
"""
def factorize_nodelist(ast_node: Node):
    ""
    Detect patterns inside a ConsecutiveNode and factor them in Repeats.
    It takes a general AST Node parameter, as it makes it possible to construct a
    structural inducing version with ast_map
    ""

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
            node = (
                pattern[0] if len(pattern) == 1 else construct_consecutive_node(pattern)
            )
            if reverse:
                nnodes.append(Repeat(node, -count))
            else:
                nnodes.append(Repeat(node, count))
            i += len(pattern) * count
        else:
            nnodes.append(nodes[i])
            i += 1

    return ConsecutiveNode(nodes=nnodes)
"""

def functionalized(node: Node) -> list[tuple[Node, Any]]:
    """
    Return a functionalized version of the node and a parameter.
    As the parameter count of Repeat is not a ASTNode, it's functionalized version
    is 0 to mark it needs to be replaced, -index once replaced
    """
    match node:
        # chained function calls
        # For now only output two-chains, but should also propose n-chains if they exist
        # add s_x(s_y(s_z())) add nauseam until the pattern
        # case SymbolicNode(i, parameters, l) if False and len(parameters) == 1 and not isinstance(parameters, Variable):
        #    results = []
        #    # HAVING A VARIABLE IN PLACE OF PARAMETERS AND DOING THIS IS GIGA BUGGY AT REPLACEMENT AT THE MOMENT
        #    if False and isinstance(parameters[0], SymbolicNode):
        #        n = parameters[0]
        #        results.append((SymbolicNode(i, (SymbolicNode(n.index, Variable(0), n.len_ref),),l), (n.parameters, )))
        #    return results
        # TO DO: One day, if a sequence that already contain a variable is functionalized
        # Add more index to the variable and uncurrify it
        case SequenceNode(nodes=nodes):

            def construct_sequence(nodes):
                return (
                    construct_consecutive_node(nodes)
                    if isinstance(node, ConsecutiveNode)
                    else AlternativeNode(nodes)
                )

            CUTOFF_PURE_NODE_REPLACEMENT = 2  # A 2 sequence node with 2 parameters is like memorizing the 2sequences

            if any(isinstance(n, Variable) for n in nodes):
                return []

            node_set = set(nodes)
            max_nodes = sorted(node_set, key=len, reverse=True)[:2]

            nodes1 = [
                nnode if nnode != max_nodes[0] else Variable(0) for nnode in nodes
            ]
            result1 = (construct_sequence(nodes1), (max_nodes[0],))

            if len(max_nodes) > 1 and len(nodes) > CUTOFF_PURE_NODE_REPLACEMENT:
                nodes2 = [
                    Variable(max_nodes.index(n)) if n in max_nodes else n for n in nodes
                ]
                result2 = (construct_sequence(nodes2), tuple(max_nodes))

                return [result1, result2]
            else:
                return [result1]
        case Repeat(node=nnode, count=count):
            functions = []
            if not isinstance(nnode, Variable) and not isinstance(count, Variable):
                functions = [
                    (Repeat(Variable(0), count), (nnode,)),
                    (Repeat(nnode, Variable(0)), (count,)),
                ]
            return functions
        case Root(start=s, colors=c, node=n):
            functions = []
            if (
                not isinstance(s, Variable)
                and not isinstance(n, Variable)
                and not isinstance(c, Variable)
            ):
                functions = [
                    # Bar the Root to memorize position otherwise it defeats the purpose of "objectification"
                    (Root(Variable(0), c, n), (s,)),
                    # (Root(s, c, Variable(0)), n),
                    (Root(Variable(0), c, Variable(1)), (s, n)),
                ]
                if len(c) == 1:
                    functions.extend(
                        [
                            # (Root(s, Variable(0), n), c),
                            (Root(Variable(0), Variable(1), n), (s, c)),
                            # (Root(s, Variable(0), Variable(1)), [c, n])
                        ]
                    )
            return functions
        case Rect(height=h, width=w) if h == w and not isinstance(h, Variable):
            return [(Rect(Variable(0), Variable(0)), (h,))]
        case Rect(height=h, width=w) if not isinstance(h, Variable) and not isinstance(
            w, Variable
        ):
            return [(Rect(Variable(0), w), (h,)), (Rect(h, Variable(0)), (w,))]
        case _:
            return []


### Other helper functions
### Main functions
def update_node_i_by_j(node: Node, i, j):
    match node:
        case SymbolicNode(index, param, l) if index == i:
            return SymbolicNode(j, param, l)
        case _:
            return node

    # return node.copy()


def update_node(node, mapping):
    match node:
        case SymbolicNode(i, parameters, l):
            if DEBUG_ASTMAP:
                print("\n")
                print(f"Mapping node: {node}")
                print(f"to {SymbolicNode(mapping[i], parameters, l)}")
                print(f"With the mapping: {i} -> {mapping[i]}")
            # if isinstance(param, ASTNode):
            #    param = ast_map(lambda node: update_node(node, mapping), param)
            return SymbolicNode(mapping[i], param, l)

    return node


def construct_node1(
    coordinates,
    is_valid: Callable[[Coord], bool],
    traversal: TraversalModes = TraversalModes.BFS,
):
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

    node = bfs(coordinates) if traversal == TraversalModes.BFS else dfs(coordinates)
    return ast_map(extract_rects, ast_map(factorize_nodelist, node))

def freeman_to_ast(freeman_node: FreemanNode) -> Optional[Node]:
    branches = []
    nodelist = []

    for branch in freeman_node.children:
        node = freeman_to_ast(branch)
        # if isinstance(node, AlternativeNode):
        #    node = AlternativeNode(get_iterator(node.nodes))#encode_run_length(node)
        branches.append(node)

    # And we compute the main path
    if len(freeman_node.path) > 0:
        if GREEDY:
            path = encode_run_length(
                MovesNode("".join([str(move) for move in freeman_node.path]))
            )
        else:
            path = "".join([str(move) for move in freeman_node.path])
            rect = extract_rects(MovesNode(path))
            if rect == MovesNode(path):
                # path = encode_string(path)
                path = dynamic_factorize(path)
            else:
                path = rect
        nodelist.append(path)

    branches = branch_from_list(branches)
    if branches is not None:
        nodelist.append(branches)

    astnode = node_from_list(nodelist)
    return astnode


def freeman_to_ast_Leaf(freeman_node: FreemanNode) -> Optional[Node]:
    branches = []
    nodelist = []

    for branch in freeman_node.branches:
        node = freeman_to_ast(branch)
        branches.append(node)

    # And we compute the main path
    if len(freeman_node.path) > 0:
        path = [Leaf(Move(move)) for move in freeman_node.path]
        rect = extract_rects(path)
        if is_leaf_list(rect):
            path = dynamic_factorize(path)
        else:
            path = rect
        nodelist.append(path)

    branches = branch_from_list(branches)
    if branches is not None:
        nodelist.append(branches)

    astnode = node_from_list(nodelist)
    return astnode


def construct_node(freeman_node: FreemanNode) -> Optional[Node]:
    # Recursive part: first we get the nodes from the different branches
    astnode = freeman_to_ast(freeman_node)
    if GREEDY:
        return ast_map(
            extract_rects, ast_map(factorize_nodelist, astnode)
        )  # ast_map(extract_rects, ast_map(factorize_nodelist, astnode))
    else:
        # return ast_map(factorize_nodelist, astnode)
        return astnode


### Symbolic System
def replace_parameters(node: Node, parameters: tuple):
    # Hypothesis: the parameters here are either not ASTNodes, or not symbolic
    if isinstance(node, Variable):
        if node.index < 0 or node.index >= len(parameters):
            raise IndexError(
                f"Parameters: {parameters}, Bad Variable index {node.index}, params has {len(parameters)} items"
            )
        return parameters[node.index]

    # For everything not node, when replace_parameters is called recursively
    if not isinstance(node, Node):
        return node

    ## BUGGY PART
    # if False and isinstance(node, SymbolicNode):
    #    new_params = node.parameters
    #    if isinstance(new_params, Variable):
    #        new_params = replace_parameters(new_params, parameters)
    #    elif isinstance(new_params, tuple):
    #        new_params = tuple(replace_parameters(param, parameters) for param in new_params)
    #    return SymbolicNode(node.index, new_params, node.len_ref)

    new_attrs = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, (list, tuple)):
            try:
                new_attrs[f.name] = type(value)(
                    replace_parameters(item, parameters) for item in value
                )
            except:
                print(f"Node: {node}, parameters: {parameters}")
        else:
            new_attrs[f.name] = replace_parameters(value, parameters)

    return type(node)(**new_attrs)


SymbolTable = list[Node]


def symbolize_next(
    ast_ls: list[Node],
    refs: SymbolTable,
    lattice_count=1,
    include_functionalized=False,
    include_roots=False,
) -> tuple[list[ASTNode], SymbolTable, bool]:
    # co_symbolize a list of ast
    pattern_counter = Counter()
    pattern_metadata = {}

    def register_node(pattern, param=None):
        # pattern = (count, value, value_param)
        pattern_counter[pattern] += 1
        if pattern not in pattern_metadata:
            pattern_metadata[pattern] = (len(pattern), len_param(param))
        else:
            _, value_param = pattern_metadata[pattern]
            pattern_metadata[pattern] = (len(pattern), value_param + len_param(param))

    def discover_node(node: ASTNode):
        """
        Add a node to the dictionary of node, and its functionalized variants
        """
        if isinstance(node, (Variable, UnionNode)):
            return

        if not isinstance(
            node, (SymbolicNode, Variable, Root, UnionNode)
        ):  # the not Variable is probably not useful
            register_node(node)

        # TRY TO PUT THE FUNCTIONALIZED IN THE PREVIOUS
        # ESPECIALY, SYMBOLIC NODE CAN BE IN THEORY MEMORIZED HERE, SHOULDNT
        if (include_functionalized or not isinstance(node, SymbolicNode)) and (
            include_roots or not isinstance(node, Root)
        ):
            for fun, parameter in functionalized(node):
                register_node(fun, parameter)

        for child in children(node):
            discover_node(child)

    # For each ast and templates already in the symbol table, discover every node

    for node in ast_ls + list(refs):
        if node is not None:
            discover_node(node)

    # 6: cost of a symbolic node
    def bit_gained(node):
        value, _ = pattern_metadata[node]
        count = pattern_counter[node]
        value_symb = len(SymbolicNode(-1, (), 0))
        return count * value - (count - 1) * value_symb - value

    compressable = [
        (node, gain)
        for node in pattern_metadata
        if (gain := bit_gained(node)) > 0
        and pattern_counter[node] > lattice_count
        and node not in refs
    ]

    if not compressable:
        return ast_ls, refs, False

    max_symb, _ = max(compressable, key=lambda x: x[1])
    symbol_index = len(refs)

    def replace_by_symbol(node: ASTNode) -> ASTNode:
        # If the symbol is a constant, and equal to the current node repl
        # Replace the current node by the symbol
        if node == max_symb:
            return SymbolicNode(symbol_index, (), len(max_symb))

        # Else, propragate the symbolic node
        match node:
            case SymbolicNode(index, parameters, len_ref) if not isinstance(
                parameters, Variable
            ):
                new_params = tuple(
                    replace_by_symbol(p) if isinstance(p, ASTNode) else p
                    for p in parameters
                )
                return SymbolicNode(index, new_params, len_ref)
            case SequenceNode(nodes=nodes):
                node = type(node)([replace_by_symbol(n) for n in nodes])
            case Repeat(node=n, count=c):
                node = Repeat(replace_by_symbol(n), c)
            case Root(start=s, colors=c, node=n):
                node = Root(s, c, replace_by_symbol(n))

        # And test for functions
        for fun, parameter in functionalized(node):
            if fun == max_symb:
                return SymbolicNode(symbol_index, parameter, len(max_symb))

        return node

    # Replace symbols in the main AST list
    new_ast_ls = [replace_by_symbol(node) for node in ast_ls]

    refs = [replace_by_symbol(ref) for ref in refs]
    refs.append(max_symb)  # Simplified add_symbol

    if 12 <= len(refs) <= 14:
        print("refs:")
        for ref in refs:
            print(ref)

        print("asts:")
        for ast in new_ast_ls:
            print(ast)

    # Note you can only replace one at a time because a same node can be part of several 1-form
    return new_ast_ls, refs, True


@handle_elements
def symbolize(
    ast_ls: list[Node], refs: SymbolTable, lattice_count=1
) -> tuple[list[Node], SymbolTable]:
    """Symbolize ast_ls as much as possible"""

    # First symbolize everything that doesn't countains symbols
    ast_ls, refs, changed = symbolize_next(
        ast_ls, refs, lattice_count=1, include_functionalized=False
    )
    while changed:
        ast_ls, refs, changed = symbolize_next(
            ast_ls, refs, lattice_count=1, include_functionalized=False
        )

    # Then symbolize also things that contains symbols
    # TO-DO: By level of symbolization
    ast_ls, refs, changed = symbolize_next(
        ast_ls, refs, lattice_count=1, include_functionalized=True
    )
    while changed:
        ast_ls, refs, changed = symbolize_next(
            ast_ls, refs, lattice_count=1, include_functionalized=True
        )

    # Then include roots:
    ast_ls, refs, changed = symbolize_next(ast_ls, refs, 1, True, True)
    while changed:
        ast_ls, refs, changed = symbolize_next(ast_ls, refs, 1, True, True)

    print("\n Refs:")
    for ref in refs:
        print(ref)

    print("\n ASTs")
    for ast in ast_ls:
        print(ast)

    return ast_ls, refs


# TO-DO: try to consolidate the symbol table through sub-node symbolization. It could use ast_distance:
# Example:
# Symbol n°23: Var(1)->Var(0): s_6( s_8(2))
# Symbol n°24: Var(1)->Var(0): s_6( s_8( s_21 ))
# The s6( s8(x)) might be symbolizable

# Or
# Symbol n°5:  s_13( s_2( s_12 ))
# Symbol n°6:  s_13( s_2( s_0(3)))
# Symbol n°7:  s_13( s_2( s_0(5)))
# Symbol n°8:  s_13( s_2( s_0(6)))
# Could be symbolized in Var


def resolve_symbolic(node: Node, symbol_table: SymbolTable) -> Any:
    if isinstance(node, SymbolicNode):
        template = symbol_table[node.index]
        params = node.parameters
    elif isinstance(node, SequenceNode):
        resolved_nodes = [resolve_symbolic(n, symbol_table) for n in node.nodes]
        return type(node)(nodes=resolved_nodes)
    elif not isinstance(node, Node):
        return node
    else:
        new_attrs = {
            attr: resolve_symbolic(value, symbol_table)
            for attr, value in node.__dict__.items()
        }
        return type(node)(**new_attrs)

    # Resolve each parameters in case they are themselves symbol
    # It means applying the lambda expression recursively
    # if isinstance(params, Variable):
    #    print(node)
    #    return replace_parameters(template, params)
    # else:
    params_resolved = tuple(
        resolve_symbolic(p, symbol_table) if isinstance(p, ASTNode) else p
        for p in params
    )
    return replace_parameters(template, params_resolved)


@handle_elements
def unsymbolize(ast_ls: list[Node], refs: SymbolTable):
    """
    Unsymbolize an AST: eliminates all Symbolic Nodes.

    1. First eliminates Symbolic Nodes from the symbol table refs: expand each symbol
    2. The Replace each SymbolicNode within each AST
    """
    # Copy the symbol table
    # refs = [copy_ast(ref) for ref in refs]
    crefs = refs

    # Define the resolve function on the copy
    def resolve(node):
        resolved = resolve_symbolic(node, crefs)
        # if is_symbolic(resolved):
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
    # crefs = [resolve(ref) for ref in crefs]

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


def factor_by_refs(node: Node, refs: SymbolTable):
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
            return SymbolicNode(ref_index, (), len(refs[ref_index]))
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


def map_refs(
    refs: SymbolTable, refs_common: SymbolTable
) -> tuple[list[int], SymbolTable]:
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
                    ref_updated = ast_map(
                        lambda node: update_node_i_by_j(node, ss, mapping_ref[ss]),
                        ref_updated,
                    )
                # The check if it already exist in the common symbol table
                # If yes then reference the new index
                if ref_updated in refs_common:
                    mapping_ref[i] = refs_common.index(ref_updated)
                # Else update it's index and add it to the common table
                else:
                    mapping_ref[i] = len(refs_common)
                    ref_updated = ast_map(
                        lambda node: update_symbol(node, mapping_ref[i]), ref_updated
                    )
                    refs_common.append(ref_updated)
    mapping = sorted([(i, j) for i, j in mapping_ref.items()], key=lambda x: x[0])
    mapping = [x[1] for x in mapping]
    return mapping, refs_common


def fuse_refs(refs_ls: list[SymbolTable]) -> tuple[SymbolTable, list[int]]:
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
                # ref_updated = ast_map(lambda node: update_node(node, mapping), ref)
                # refs_common.append(ref_updated)
            mapping_ref[ref] = nindex

        #
        # mapping.append(nindex)
        # First find the mapping then update the nodes
        return reversed(mapping)

    nrefs = []
    mappings = []
    for refs in nrefs_ls:
        # Map the new symbol table and fill with placeholders
        mapping, nrefs = map_refs(refs, nrefs)
        mappings.append(mapping)
    return nrefs, mappings


def update_asts(ast_ls: list[Node], nrefs: SymbolTable, mapping: list[int]):
    if DEBUG_ASTMAP:
        nast_ls = []
        print("\n")
        for i, ast in enumerate(ast_ls):
            print("----------------------")
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


def construct_union(
    code: Optional[Node],
    codes: list[Node],
    unions: list[Node],
    refs: SymbolTable,
    box: Box,
):
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
    background = unsymbolize(code, refs)  # type: ignore

    # Expel the unions from codes
    code_unions = set((ncode for ncode in unsymbolized if isinstance(ncode, UnionNode)))
    code_roots = set((ncode for ncode in unsymbolized if isinstance(ncode, Root)))

    # fuse code_unions with unions
    code_unions.update(unions)
    codes_dump = set()

    # codes = [c for c in codes if c is not None]
    # codes.sort(reverse=True, key=lambda x: len(x[1]))
    def remove_symbolized_code(
        input: dict[Color, set[tuple["ASTNode", "ASTNode"]]],
    ) -> dict[Color, set["ASTNode"]]:
        codes = {}
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
        if any(codes_count[ncode] > 1 for ncode in u.codes) or (
            u.shadowed & codes_count.keys()
        ):
            codes_dump.update(u.codes)
            codes_dump.update(u.shadowed)
        else:
            subunions.add(u)

    subcodes = defaultdict(set)

    for ncode in code_roots:
        c = next(iter(ncode.colors))  # type: ignore
        index = unsymbolized.index(ncode)
        subcodes[c].add((codes[index], ncode))
        if is_symbolic(ncode):
            print(f"From code_roots, Error: code symbolic: {ncode}")
        colors_count[c].add((codes[index], ncode))

    for ncode in codes_dump:
        c = next(iter(ncode.colors))  # type: ignore
        subcodes[c].add((factor_by_refs(ncode, refs), ncode))
        if is_symbolic(ncode):
            print(f"From codes_dump: Error: code symbolic: {ncode}")
        colors_count[c].add((factor_by_refs(ncode, refs), ncode))

    for u in code_unions:
        for ncode in u.codes:
            c = next(iter(ncode.colors))  # type: ignore
            colors_count[c].add((factor_by_refs(ncode, refs), ncode))
        c = next(iter(u.background.colors))  # type: ignore
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
            len_color_symbolic += (
                0 if not isinstance(symbolized, SymbolicNode) else symbolized.len_ref
            )
        len_colors.append((color, len_color, len_color_symbolic, is_border))

    # / 5 is an evil arbitrary parameter necessary in case some node memorization is "too" efficient
    len_colors.sort(reverse=True, key=lambda x: (x[3], x[1] + x[2] / 5))

    nbackground = None
    nshadowed = None
    nsubunions = set()
    unions_to_remove = []

    if len_colors:
        color, length, len_ref, is_border = len_colors[0]
        if background and is_border and length + len_ref / 5 > len(code):
            # background.colors = {color}
            nbackground = Root(background.start, {color}, background.node)  # background

            for union in subunions:
                if next(iter(union.background.colors)) == color or (
                    union.codes & subcodes[color]
                ):
                    for code in union.codes:
                        c = next(iter(code.colors))  # type: ignore
                        subcodes[c].add((code, code))
                        unions_to_remove.append(union)
                    # for code in union.shadowed:
                    #    c = next(iter(code.colors)) # type: ignore
                    #    subcodes[c].add((code, code))
            nshadowed = set((unsymbolized for ncode, unsymbolized in subcodes[color]))
            del subcodes[color]

    for union in subunions:
        if union not in unions_to_remove:
            nsubunions.add(union)

    if subcodes:
        subcodes = quotient_to_set(remove_symbolized_code(input=subcodes))
    else:
        subcodes = set()

    return UnionNode(nsubunions | subcodes, nshadowed, nbackground)  # type: ignore


def shift_ast(shift: Coord, node: Node):
    def displace(node):
        match node:
            case Root(s, c, n) if isinstance(s, tuple):
                return Root((s[0] + shift[0], s[1] + shift[1]), c, n)
            case SymbolicNode(i, p, l) if isinstance(p, tuple):
                return SymbolicNode(i, (p[0] + shift[0], p[1] + shift[1]), l)
        return node

    return ast_map(displace, node)


@optional
def decode(
    node: Node, coordinates: Coord = (0, 0), color: int = 1
) -> tuple[Coord, Points]:
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
        raise ValueError(
            f"Trying to decode a node {node} containing symbols. Please resolve its symbols first"
        )

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
                if (col, row) not in coords:
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
        case Leaf(data) if isinstance(data, Move):
            col, row = ncoordinates
            move = data.value

            col += DIRECTIONS_FREEMAN[move][0]
            row += DIRECTIONS_FREEMAN[move][1]

            points.append((col, row, color))
            ncoordinates = (col, row)
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
        case None:
            pass
        case _:
            raise ValueError(f"Unexpected node when decoding: {node}")

    return ncoordinates, set(points)


def node_to_grid(node: Node) -> Grid:
    _, points = decode(node)
    return points_to_grid_colored(points)


def ast_distance1(
    node1: Optional[Node], node2: Optional[Node], refs: SymbolTable
) -> int:
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
                if list1[i - 1] == list2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Calculate the cost of each operation
                    replace_cost = dp[i - 1][j - 1] + ast_distance(
                        list1[i - 1], list2[j - 1], refs
                    )
                    delete_cost = dp[i - 1][j] + len(list1[i - 1])
                    insert_cost = dp[i][j - 1] + len(list2[j - 1])

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
        total_cost = sum(len(ast) for ast in only_in_s1 if ast) + sum(
            len(ast) for ast in only_in_s2 if ast
        )

        if only_in_s1 and only_in_s2:
            # Create edit matrix with tuples: (edit_distance, s1_length, s2_length)
            edit_matrix = [
                [
                    (ast_distance(a, b, refs), len(a) if a else 0, len(b) if b else 0)
                    for b in only_in_s2
                ]
                for a in only_in_s1
            ]

            while edit_matrix and any(
                edit_matrix
            ):  # Check if matrix is not empty and has non-empty rows
                # Find minimum edit distance
                min_dist, s1_len, s2_len = max(
                    (item for row in edit_matrix for item in row if item),
                    key=lambda x: x[1] + x[2] - x[0],
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
        return min_dist + sum([len(n) for n in (ls[:min_i] + ls[min_i + 1 :])])

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
                raise ValueError(f"NoneCase wrongly called on {node1}, {node2}")

    match node1, node2:
        case (None, _) | (_, None):
            return handle_none_cases(node1, node2)
        case (UnionNode(c1, s1, b1), UnionNode(c2, s2, b2)):
            # len_back = ast_distance(b1, b2, refs)
            len_codes = set_edit_distance(c1 | {b1}, c2 | {b2})
            return len_codes  # len_back + len_codes
        case (UnionNode(c, s, b), _):
            len_b = len(b) if b else 0
            return min(
                node_list_distance(node2, c) + len_b,
                sum([len(n) for n in c]) + ast_distance(node2, b, refs),
            )
        case (_, UnionNode(c, s, b)):
            return ast_distance(node2, node1, refs)
        case (SymbolicNode(i1, p1, l1), SymbolicNode(i2, p2, l2)):
            if i1 == i2:
                return sum([param_distance(p1[i], p2[i]) for i in range(len(p1))])
            else:
                return ast_distance(
                    unsymbolize(node1, refs), unsymbolize(node2, refs), refs
                )  # type :ignore
        case (SymbolicNode(i, parameters, l), _):
            # Calculate distance if we unsymbolize the node
            unsymbolized_distance = ast_distance(unsymbolize(node1, refs), node2, refs)  # type: ignore

            # Calculate distances for each parameter
            param_distances = [
                (
                    i,
                    ast_distance(p, node2, refs)
                    if isinstance(p, ASTNode)
                    else len(node2) + len_param(p),
                )
                for i, p in enumerate(parameters)
            ]

            symbolic_distance = BitLength.NODE + BitLength.INDEX
            # Find parameter with minimum distance
            if param_distances:
                min_index, min_param_distance = min(param_distances, key=lambda x: x[1])
                # Calculate symbolic distance
                symbolic_distance += sum(len(parameters[i]) for i in range(min_index))
                symbolic_distance += min_param_distance
                symbolic_distance += sum(
                    len(parameters[i]) for i in range(min_index + 1, len(parameters))
                )
            else:
                symbolic_distance += len(node2)

            return min(unsymbolized_distance, symbolic_distance)
        case (_, SymbolicNode(i, p, l)):
            return ast_distance(node2, node1, refs)
        case (Root(s1, c1, n1), Root(s2, c2, n2)):
            return (
                abs(s1[0] - s2[0])
                + abs(s1[1] - s2[1])
                + len(c1 ^ c2) * BitLength.COLOR
                + ast_distance(n1, n2, refs)
            )  # type: ignore
        case (Root(s, c, n), _):
            return (
                BitLength.COORD
                + BitLength.COLOR * len(c)
                + ast_distance(n, node2, refs)
            )
        case (_, Root(s, c, n)):
            return ast_distance(node2, node1, refs)
        case (AlternativeNode(nnodes=nls1), AlternativeNode(nnodes=nls2)):
            return set_edit_distance(nls1, nls2)
        case (AlternativeNode(nnodes=nls), _):
            return BitLength.NODE + node_list_distance(node2, nls)
        case (_, AlternativeNode(nnodes=nls)):
            return ast_distance(node2, node1, refs)
        case (Rect(h1, w1), Rect(h2, w2)):
            return 0 if h1 == h2 and w1 == w2 else BitLength.COORD
        case (Rect(h, w), _):
            # return min(ast_distance(node2, rect_to_moves(h, w), refs), LEN_COORD+len(node2))
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
        case (Repeat(n, c), _):
            dist = BitLength.NODE + BitLength.COUNT + ast_distance(n, node2, refs)
            return dist
        case (_, Repeat(n, c)):
            return ast_distance(node2, node1, refs)
        case (MovesNode(m1), MovesNode(m2)):
            return distance_levenshtein(m1, m2) * BitLength.MOVE
        case _:
            return 0 if node1 == node2 else len(node1) + len(node2)


def ast_distance(
    node1: Optional[Node], node2: Optional[Node], refs: SymbolTable
) -> int:
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
                if list1[i - 1] == list2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Calculate the cost of each operation
                    replace_cost = dp[i - 1][j - 1] + ast_distance(
                        list1[i - 1], list2[j - 1], refs
                    )
                    delete_cost = dp[i - 1][j] + len(list1[i - 1])
                    insert_cost = dp[i][j - 1] + len(list2[j - 1])

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
        total_cost = sum(len(ast) for ast in only_in_s1 if ast) + sum(
            len(ast) for ast in only_in_s2 if ast
        )

        if only_in_s1 and only_in_s2:
            # Create edit matrix with tuples: (edit_distance, s1_length, s2_length)
            edit_matrix = [
                [
                    (ast_distance(a, b, refs), len(a) if a else 0, len(b) if b else 0)
                    for b in only_in_s2
                ]
                for a in only_in_s1
            ]

            while edit_matrix and any(
                edit_matrix
            ):  # Check if matrix is not empty and has non-empty rows
                # Find minimum edit distance
                min_dist, s1_len, s2_len = max(
                    (item for row in edit_matrix for item in row if item),
                    key=lambda x: x[1] + x[2] - x[0],
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
        return min_dist + sum([len(n) for n in (ls[:min_i] + ls[min_i + 1 :])])

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
                raise ValueError(f"NoneCase wrongly called on {node1}, {node2}")

    match node1, node2:
        case (None, _) | (_, None):
            return handle_none_cases(node1, node2)
        case (UnionNode(c1, s1, b1), UnionNode(c2, s2, b2)):
            # len_back = ast_distance(b1, b2, refs)
            len_codes = set_edit_distance(c1 | {b1}, c2 | {b2})
            return len_codes  # len_back + len_codes
        case (UnionNode(c, s, b), _):
            len_b = len(b) if b else 0
            return min(
                node_list_distance(node2, c) + len_b,
                sum([len(n) for n in c]) + ast_distance(node2, b, refs),
            )
        case (_, UnionNode(c, s, b)):
            return ast_distance(node2, node1, refs)
        case (SymbolicNode(i1, p1, l1), SymbolicNode(i2, p2, l2)):
            if i1 == i2:
                return sum([param_distance(p1[i], p2[i]) for i in range(len(p1))])
            else:
                return ast_distance(
                    unsymbolize(node1, refs), unsymbolize(node2, refs), refs
                )  # type :ignore
        case (SymbolicNode(i, parameters, l), _):
            # Calculate distance if we unsymbolize the node
            unsymbolized_distance = ast_distance(unsymbolize(node1, refs), node2, refs)  # type: ignore

            # Calculate distances for each parameter
            param_distances = [
                (
                    i,
                    ast_distance(p, node2, refs)
                    if isinstance(p, ASTNode)
                    else len(node2) + len_param(p),
                )
                for i, p in enumerate(parameters)
            ]

            symbolic_distance = BitLength.NODE + BitLength.INDEX
            # Find parameter with minimum distance
            if param_distances:
                min_index, min_param_distance = min(param_distances, key=lambda x: x[1])
                # Calculate symbolic distance
                symbolic_distance += sum(len(parameters[i]) for i in range(min_index))
                symbolic_distance += min_param_distance
                symbolic_distance += sum(
                    len(parameters[i]) for i in range(min_index + 1, len(parameters))
                )
            else:
                symbolic_distance += len(node2)

            return min(unsymbolized_distance, symbolic_distance)
        case (_, SymbolicNode(i, p, l)):
            return ast_distance(node2, node1, refs)
        case (Root(s1, c1, n1), Root(s2, c2, n2)):
            return (
                abs(s1[0] - s2[0])
                + abs(s1[1] - s2[1])
                + len(c1 ^ c2) * BitLength.COLOR
                + ast_distance(n1, n2, refs)
            )  # type: ignore
        case (Root(s, c, n), _):
            return (
                BitLength.COORD
                + BitLength.COLOR * len(c)
                + ast_distance(n, node2, refs)
            )
        case (_, Root(s, c, n)):
            return ast_distance(node2, node1, refs)
        case (AlternativeNode(nnodes=nls1), AlternativeNode(nnodes=nls2)):
            return set_edit_distance(nls1, nls2)
        case (AlternativeNode(nnodes=nls), _):
            return BitLength.NODE + node_list_distance(node2, nls)
        case (_, AlternativeNode(nnodes=nls)):
            return ast_distance(node2, node1, refs)
        case (Rect(h1, w1), Rect(h2, w2)):
            return 0 if h1 == h2 and w1 == w2 else BitLength.COORD
        case (Rect(h, w), _):
            # return min(ast_distance(node2, rect_to_moves(h, w), refs), LEN_COORD+len(node2))
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
        case (Repeat(n, c), _):
            dist = BitLength.NODE + BitLength.COUNT + ast_distance(n, node2, refs)
            return dist
        case (_, Repeat(n, c)):
            return ast_distance(node2, node1, refs)
        case (MovesNode(m1), MovesNode(m2)):
            return distance_levenshtein(m1, m2) * BitLength.MOVE
        case _:
            return 0 if node1 == node2 else len(node1) + len(node2)
