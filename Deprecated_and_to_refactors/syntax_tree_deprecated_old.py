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
    KNode,
    BitLengthAware,
    KolmogorovTree,
    ProductNode,
    SumNode,
    RepeatNode,
    SymbolNode,
    RootNode,
    SymbolNode,
    PrimitiveNode,
    MoveValue,
    PaletteValue,
    IndexValue,
    VariableValue,
    CoordValue,
    BitLength,
    resolve_symbols,
    symbolize,
    children,
    breadth_iter,
    reverse_node,
    encode_run_length,
    is_function,
    contained_symbols,
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
@dataclass
class UnionNode:
    """
    Represent a connected component by reconstructing it with the best set of single color programs.
    After marginalisation comes reconstruction, divide and conquer.
    """

    nodes: set[KNode]
    shadowed: set[int] | None = None
    normalizing_node: KNode | None = None

    def bit_length(self) -> int:
        # Maybe remove the cost of the shadowed nodes
        len_nodes = sum(node.bit_length() for node in self.nodes)
        return len_nodes + (
            0 if self.normalizing_node is None else self.normalizing_node.bit_length()
        )


@dataclass()
class UnionNode1:
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
            msg += f"{self.background} < "
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
"""

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

### Other helper functions
### Main functions
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


# TO Refactor
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
