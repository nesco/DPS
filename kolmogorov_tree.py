"""
Kolmogorov Tree: An abstract-syntax tree structure forming to represent non-deterministic bit-length-aware programs.
TL;DR: An bitlengthaware AST for bitlengthaware rosetrees

This module implements a tree structure for representing non-deterministic programs,
for example to describe shapes using a non-deterministic program representing 2D grid movements (see syntax_tree.py). The representation
aims to approximate Kolmogorov complexity through minimum description length.

The tree structure consists of:
- ProductNodes for deterministic sequences
- SumNodes for non-deterministic branching
- RepeatNodes for repetition extraction
- SymbolNodes for pattern abstraction and reuse
- Variable binding for lambda abstraction

Key Features:
- Computable bit-length metrics for complexity approximation
- Pattern extraction and memorization via symbol table
- Support for 8-directional grid movements
- "lambda abstraction" through variable binding flattened into functions with arity
- Nested pattern detection and reuse

The tree can be used to:
1. Represent shapes as non-deterministic programs
2. Extract common patterns across multiple programs
3. Measure and optimize program complexity
4. Enable search over program space guided by complexity


In a way, BitLengthAware is a type encapsulation, so the types "know" their bit length, like types that knows
their gradients in deep learning.

Example Usage:
```python
# Create a simple pattern
pattern = create_moves_sequence("2323")  # Right-Down-Right-Down
# Repeat it 3 times
repeated = RepeatNode(pattern, 3)
# Add non-deterministic branching
program = SumNode([repeated, create_rect(3, 3)])
# Create full tree with starting position and colors
tree = KolmogorovTree(RootNode((0,0), {1}, program))
```

Non-trial combinations:
- SumNode({RepeatNode()}) <- single element, which is a repeat, that get transformed into an iterator, otherwise for single elements SumNodes might need to get flattened
- RepeatNode(SumNode, count) <- should be nested alternatives
- RootNode should either be top level, or in a SumNode containing only RootNodes
"""

# TO-DO:
# Usually when constructed through a graph DFS, sum nodes always ends-up at the end of a product node if in a sequence
# It could lead to further compression, if the end of the sum node repeats
# E.G 111 [134,244,334] -> 111[13, 24, 33]4

import functools
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Collection
from dataclasses import dataclass, field, fields, is_dataclass
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Self,
    Sequence,
    TypeVar,
    cast,
)

from edit import (
    Add,
    Delete,
    Operation,
)
from localtypes import (
    BitLengthAware,
    Color,
    Coord,
    Primitive,
    ensure_all_instances,
)
from tree_functionals import (
    breadth_first_preorder,
    dataclass_subvalues,
    depth_first_preorder,
    postorder_map,
    preorder_map,
)

# Type variable for generic programming


# Bit length definitions
class BitLength(IntEnum):
    """Defines bit lengths for generic components in the Kolmogorov Tree."""

    COUNT = 5  # 5 bits for repeat counts (0-31), specific value suitable for ARC grid sizes
    NODE_TYPE = 3  # 3 bits for up to 8 node types
    INDEX = 7  # 7 bits for symbol indices (up to 128 symbols)
    VAR = 1  # 2 bits for variable indices (up to 2 variables per symbol)


class ARCBitLength(IntEnum):
    """Defines bit lengths for components tailored for ARC AGI."""

    COORD = 10  # 10 bits for coordinates (5 bits per x/y, for 0-31)
    COLORS = 4  # 4 bits for primitives (0-9)
    DIRECTIONS = 3  # 3 bits for primitives (directions 0-7)


@dataclass(frozen=True)
class Alphabet(Primitive):
    """Base class for all types which are program outputs. The generic type 'T' is bound to it. It offers a shifting operation"""

    @abstractmethod
    def shift(self, k: int) -> Self:
        """Shifts the primitive value by k steps."""
        pass

    @staticmethod
    def size() -> int:
        """Size of the alphabet"""
        return 0


@dataclass(frozen=True)
class CountValue(Primitive):
    """Represent a 5-bit int"""

    value: int

    def bit_length(self) -> int:
        return BitLength.COUNT


@dataclass(frozen=True)
class VariableValue(Primitive):
    """Represents a variable as an index (0-3)."""

    value: int

    def bit_length(self) -> int:
        return BitLength.VAR


@dataclass(frozen=True)
class IndexValue(Primitive):
    """Represents an index in the lookup table (0-127)."""

    value: int

    def bit_length(self) -> int:
        return BitLength.INDEX  # 7 bits


# ARC Specific primitves
@dataclass(frozen=True)
class MoveValue(Alphabet):
    """Represents a single directional move (0-7 for 8-connectivity)."""

    value: int

    def bit_length(self) -> int:
        return ARCBitLength.DIRECTIONS  # 4 bits

    def shift(self, k: int) -> "MoveValue":
        return MoveValue((self.value + k) % 8)

    @staticmethod
    def size():
        return 8


@dataclass(frozen=True)
class PaletteValue(Primitive):
    """Represents a color value (0-9 in ARC AGI)."""

    value: frozenset[Color]

    def bit_length(self) -> int:
        return ARCBitLength.COLORS * len(self.value)  # 4 bits

    def __str__(self):
        return f"{set(self.value)}"


@dataclass(frozen=True)
class CoordValue(Primitive):
    """Represents a 2D coordinate pair."""

    value: Coord

    def bit_length(self) -> int:
        return ARCBitLength.COORD  # 10 bits (5 per coordinate)


T = TypeVar("T", bound=Alphabet)


# Base node class
@dataclass(frozen=True)
class KNode(Generic[T], BitLengthAware, ABC):
    def __len__(self) -> int:
        return self.bit_length()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def bit_length(self) -> int:
        return BitLength.NODE_TYPE

    def __or__(self, other: "KNode[T]") -> "SumNode[T]":
        """Overloads | for alternatives, unpacking SumNodes."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        children = []
        if isinstance(self, SumNode):
            children.extend(self.children)
        else:
            children.append(self)
        if isinstance(other, SumNode):
            children.extend(other.children)
        else:
            children.append(other)
        return SumNode(frozenset(children))

    def __and__(self, other: "KNode[T]") -> "ProductNode[T]":
        """Overloads & for sequences, unpacking ProductNodes."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        children = []
        if isinstance(self, ProductNode):
            children.extend(self.children)
        else:
            children.append(self)
        if isinstance(other, ProductNode):
            children.extend(other.children)
        else:
            children.append(other)
        return ProductNode(tuple(children))

    def __add__(self, other: "KNode[T]") -> "ProductNode[T]":
        """Overloads + for concatenation, unpacking ProductNodes."""
        return self.__and__(other)  # Same behavior as &

    def __mul__(self, count: int) -> "RepeatNode[T]":
        """Overloads * for repetition, multiplying count if already a RepeatNode."""
        if not isinstance(count, int):
            raise TypeError("Count must be an integer")
        if count < 0:
            raise ValueError("Count must be non-negative")
        if (
            isinstance(self, RepeatNode)
            and isinstance(self.count, CountValue)
            and self.count.value * count < 2**BitLength.COUNT
        ):
            # Multiply existing count to optimize complexity
            return RepeatNode(self.node, CountValue(self.count.value * count))
        return RepeatNode(self, CountValue(count))


# Node types for program representation
@dataclass(frozen=True)
class PrimitiveNode(KNode[T]):
    """Leaf node holding a primitive value."""

    value: T

    @property
    def data(self) -> Any:
        """Returns the underlying data of the PrimitiveNode."""
        return self.value.value

    def bit_length(self) -> int:
        return super().bit_length() + self.value.bit_length()

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class VariableNode(KNode[T]):
    """Represents a variable placeholder within a symbol."""

    index: VariableValue  # Variable index (0-3 with 2 bits)

    def bit_length(self) -> int:
        return super().bit_length() + self.index.bit_length()  # 3 + 2 bits

    def __str__(self) -> str:
        return f"Var({self.index})"


@dataclass(frozen=True)
class CollectionNode(KNode[T], ABC):
    """Abstract base class for nodes with multiple children."""

    children: Collection[KNode[T]]

    def bit_length(self):
        count_bits = (
            math.ceil(math.log2(len(self.children) + 1)) if self.children else 0
        )
        return (
            super().bit_length()
            + sum(child.bit_length() for child in self.children)
            + count_bits
        )


@dataclass(frozen=True)
class ProductNode(CollectionNode[T]):
    """Represents a sequence of actions (AND operation)."""

    children: tuple[KNode[T], ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        return "".join(str(child) for child in self.children)


@dataclass(frozen=True)
class SumNode(CollectionNode[T]):
    """Represents a choice among alternatives (OR operation)."""

    children: frozenset[KNode[T]] = field(default_factory=frozenset)

    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        # Horrible hack for the already horrible iterator hack
        if len(self.children) == 1 and isinstance(
            next(iter(self.children)), RepeatNode
        ):
            return "[+" + str(next(iter(self.children))) + "]"
        # Sort children by string representation for consistent output
        sorted_children = sorted(self.children, key=str)
        return "[" + "|".join(str(child) for child in sorted_children) + "]"


@dataclass(frozen=True)
class MetaNode(KNode[T], ABC):
    """Wraps a single node with additional information. node allows None to handle edge cases for map functions"""

    node: KNode[T]

    def bit_length(self) -> int:
        return super().bit_length() + self.node.bit_length()


@dataclass(frozen=True)
class RepeatNode(MetaNode[T]):
    """Represents repetition of a node a specified number of times."""

    count: CountValue | VariableNode  # Count can be fixed or parameterized

    def bit_length(self) -> int:
        count_len = self.count.bit_length()
        return super().bit_length() + count_len

    def __str__(self) -> str:
        return f"({str(self.node)})*{{{self.count}}}"


@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """Represents an abstraction or reusable pattern."""

    index: IndexValue  # Index in the symbol table
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)
    reference_length: int = 0

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_len

    def __str__(self) -> str:
        if self.parameters:
            return (
                f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
            )
        return f"s_{self.index}"


# Dirty hack for ARC
# It will need to be an abstraction
# That will be pre-populated in the symbol table
# akin to pre-training
@dataclass(frozen=True)
class RectNode(KNode):
    height: (
        "CountValue | VariableNode"  # Use VariableNode from kolmogorov_tree.py
    )
    width: "CountValue | VariableNode"

    def bit_length(self) -> int:
        # 3 bits for node type + 8 bits for ARC-specific rectangle encoding
        height_len = (
            BitLength.COUNT
            if isinstance(self.height, int)
            else self.height.bit_length()
        )
        width_len = (
            BitLength.COUNT
            if isinstance(self.width, int)
            else self.width.bit_length()
        )
        return super().bit_length() + height_len + width_len

    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"


# Kept-here for the example, but belong's to ARC-AGI syntax tree module
# In fact, it should become "MetadataNode", as it encapsulate data that's not in the
# main target alphabet the "program" produces'
@dataclass(frozen=True)
class RootNode(MetaNode[T]):
    """Root node: it wraps the program's node with its starting context."""

    position: CoordValue | VariableNode  # Starting position
    colors: PaletteValue | VariableNode  # Colors used in the shape

    def bit_length(self) -> int:
        pos_len = self.position.bit_length()
        colors_len = self.colors.bit_length()
        return super().bit_length() + pos_len + colors_len

    def __str__(self) -> str:
        pos_str = str(self.position)
        colors_str = str(self.colors)
        node_str = str(self.node)
        return f"Root({node_str}, {pos_str}, {colors_str})"


Unsymbolized = (
    PrimitiveNode | RepeatNode | RootNode | ProductNode | SumNode | RectNode
)
Uncompressed = PrimitiveNode | ProductNode | SumNode
## Functions on KNodes


# Helpers
def shift_f(node: KNode[T], k: int) -> KNode[T]:
    if isinstance(node, PrimitiveNode) and isinstance(node.value, Alphabet):
        shifted_value = node.value.shift(k)
        return PrimitiveNode[T](shifted_value)
    return node


def next_layer(layer: Iterable[KNode]) -> tuple[KNode, ...]:
    """Used for BFS-like traversal of a K-Tree. It's basically `children` for iterable"""
    return tuple(child for node in layer for child in children(node))


type Parameters = tuple[BitLengthAware, ...]


def generate_abstractions(knode: KNode[T]) -> list[tuple[KNode[T], Parameters]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions: list[tuple[KNode[T], Parameters]] = []

    # If the node is already a lambda-abstraction
    # It should not be abstracted further
    if is_abstraction(knode):
        return abstractions

    match knode:
        case CollectionNode(children) if len(children) > 2:
            # Abstract up to two distinct elements for now
            child_counter = Counter(children)  # Count occurrences of each child
            child_set = list(child_counter.keys())
            max_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]

            # Abstract the most frequent/largest child
            nodes1 = tuple(
                VariableNode(VariableValue(0)) if c == max_children[0] else c
                for c in children
            )
            abstractions.append((type(knode)(nodes1), (max_children[0],)))

            # If there are at least two distinct children and length > 2
            if len(max_children) > 1 and len(children) > 2:
                # Abstract the second most frequent/largest child
                nodes2 = tuple(
                    VariableNode(VariableValue(0))
                    if c == max_children[1]
                    else c
                    for c in children
                )
                abstractions.append((type(knode)(nodes2), (max_children[0],)))

                # Then absttract the top two
                nodes3 = tuple(
                    VariableNode(VariableValue(max_children.index(c)))
                    if c in max_children
                    else c
                    for c in children
                )
                abstractions.append((type(knode)(nodes3), tuple(max_children)))
        case RepeatNode(node, count):
            # For a RepeatNode, either the node or the count can be abstracted
            abstractions.extend(
                [
                    (
                        RepeatNode(VariableNode(VariableValue(0)), count),
                        (node,),
                    ),
                    (
                        RepeatNode(node, VariableNode(VariableValue(0))),
                        (count,),
                    ),
                ]
            )
        case RootNode(node, position, colors):
            # In the case of a RootNode,
            # because of ARC, we don't want the position to be memorized alone
            # It would defeat the objectification lattice step'
            # TODO
            # See if the above is still applicable
            # Understand the further 'colors.value' == 1 condition
            abstractions.extend(
                [
                    # 1 parameter
                    (
                        RootNode(
                            VariableNode(VariableValue(0)), position, colors
                        ),
                        (node,),
                    ),
                    (
                        RootNode(node, VariableNode(VariableValue(0)), colors),
                        (position,),
                    ),
                    # 2 parameters
                    (
                        RootNode(
                            VariableNode(VariableValue(0)),
                            position,
                            VariableNode(VariableValue(1)),
                        ),
                        (node, colors),
                    ),
                ]
            )
            if len(colors.value) == 1:  # type: ignore -> because of 'is_abstraction', it can't be a Variable'
                abstractions.append(
                    (
                        RootNode(
                            node,
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(1)),
                        ),
                        (position, colors),
                    )
                )
        case RectNode(heigth, width):
            if heigth == width:
                abstractions.append(
                    (
                        RectNode(
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(0)),
                        ),
                        (heigth,),
                    )
                )
                abstractions.extend(
                    [
                        (
                            RectNode(VariableNode(VariableValue(0)), width),
                            (heigth,),
                        ),
                        (
                            RectNode(heigth, VariableNode(VariableValue(0))),
                            (width,),
                        ),
                    ]
                )
        case _:
            pass

    return abstractions


def get_iterator(knodes: Collection[KNode[T]]) -> frozenset[KNode[T]]:
    """
    This function identifies if the input nodes form an arithmetic sequence and encodes it as a single RepeatNode.
    It leverages a hacky double meaning: while a standard RepeatNode(X, N) means "X for _ in range(N)",
    when used as the sole child of a SumNode, e.g., SumNode((Repeat(X, N),)), it represents
    "SumNode(tuple(shift(X, k) for k in range(N)))" if N > 0, or shifts with negative increments if N < 0.
    This compresses arithmetic enumerations cost-free, enhancing expressiveness.
    """
    nodes = frozenset(knodes)
    if len(nodes) < 2:
        return frozenset(nodes)
    if all(
        isinstance(n, PrimitiveNode) and isinstance(n.value, MoveValue)
        for n in nodes
    ):
        for start_node in nodes:
            for step in [1, -1]:
                expected = {
                    shift(start_node, step * k) for k in range(len(nodes))
                }
                if expected == nodes:
                    repeat_node = RepeatNode(
                        start_node, CountValue(step * len(nodes))
                    )
                    return frozenset([repeat_node])
    return nodes


# Only useful for ARC? What if the alphabet is too large?
def get_iterator2(nodes: Iterable[KNode[T]]) -> tuple[KNode[T], ...]:
    """
    This function identifies if the input nodes form an arithmetic sequence and encodes it as a single RepeatNode.
    It leverages a hacky double meaning: while a standard RepeatNode(X, N) means "X for _ in range(N)",
    when used as the sole child of a SumNode, e.g., SumNode((Repeat(X, N),)), it represents
    "SumNode(tuple(shift(X, k) for k in range(N)))" if N > 0, or shifts with negative increments if N < 0.
    This compresses arithmetic enumerations cost-free, enhancing expressiveness.
    """
    node_ls: list[KNode[T]] = list(nodes)
    if len(node_ls) < 2:
        return tuple(node_ls)

    # Check if the sequence has a consistent increment (1 or -1)
    prev = node_ls[0]
    curr = node_ls[1]

    # Determine the increment based on the first pair
    if curr == shift(prev, 1):
        increment = 1
    elif curr == shift(prev, -1):
        increment = -1
    else:
        return tuple(node_ls)

    # Validate the entire sequence
    for i in range(2, len(node_ls)):
        prev = node_ls[i - 1]
        curr = node_ls[i]
        if curr != shift(prev, increment):
            return tuple(node_ls)

    # If we reach here, the sequence is arithmetic; encode it as a RepeatNode
    # Use increment * length to encode direction and length
    return (RepeatNode(node_ls[0], CountValue(increment * len(node_ls))),)


# Constructors


def construct_product_node(nodes: Iterable[KNode[T]]) -> ProductNode[T]:
    """
    Constructs a ProductNode from an iterable of KNode[T], applying simplifications for efficient compression.

    - Flattens nested ProductNodes to avoid unnecessary nesting.
    - Merges adjacent PrimitiveNodes into RepeatNodes where possible using run-length encoding.
    - Combines consecutive RepeatNodes with the same base node and fixed counts.
    - Preserves SumNodes as-is to maintain logical structure.

    Args:
        nodes: An iterable of KNode[T] to combine into a ProductNode.

    Returns:
        A ProductNode[T] representing the simplified sequence.
    """
    # Convert the iterable to a list for manipulation
    simplified: list[KNode[T]] = []
    for node in nodes:
        match node:
            case ProductNode(children):
                simplified.extend(children)  # Flatten nested ProductNodes
            case _:
                simplified.append(node)  # Keep other nodes as-is

    # Simplify the flattened list
    i = 0
    while i < len(simplified):
        current = simplified[i]
        match current:
            case PrimitiveNode():
                # Collect consecutive PrimitiveNodes
                primitives = [current]
                j = i + 1
                while j < len(simplified) and isinstance(
                    simplified[j], PrimitiveNode
                ):
                    primitives.append(simplified[j])  # type: ignore[reportArgumentType]
                    j += 1
                if len(primitives) > 1:
                    # Replace with run-length encoded version (assumes encode_run_length exists)
                    encoded = encode_run_length(primitives)
                    simplified[i:j] = list(
                        encoded.children
                    )  # Insert the children directly                    i += 1
                    i += len(
                        encoded.children
                    )  # Move index past inserted elements
                else:
                    i += 1
            case RepeatNode(node=base, count=CountValue(count)) if i + 1 < len(
                simplified
            ):
                next_node = simplified[i + 1]
                match next_node:
                    case RepeatNode(
                        node=next_base, count=CountValue(next_count)
                    ) if base == next_base:
                        # Combine consecutive RepeatNodes with the same base
                        combined_count = CountValue(count + next_count)
                        simplified[i] = RepeatNode(base, combined_count)
                        del simplified[i + 1]
                    case _:
                        i += 1
            case _:
                i += 1  # Move past non-mergeable nodes (e.g., SumNode)

    return ProductNode(tuple(simplified))


def iterable_to_product(iterable: Iterable[KNode[T]]) -> KNode[T] | None:
    nodes: list[KNode[T]] = list(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return nodes[0]
    else:
        return construct_product_node(nodes)


def iterable_to_sum(iterable: Iterable[KNode[T]]) -> KNode[T] | None:
    nodes: frozenset[KNode[T]] = frozenset(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return next(iter(nodes))
    else:
        return SumNode(get_iterator(nodes))


def reconstruct_knode(
    knode: KNode[T], new_children: Sequence[KNode[T]], factorize: bool = False
) -> KNode[T]:
    """Reconstructs a KNode with its original data and new children."""
    match knode:
        case CollectionNode(_):
            new_node = type(knode)(tuple(new_children))
            if factorize:
                new_node = factorize_tuple(new_node)
            return new_node
        case RepeatNode(_, count):
            return RepeatNode(new_children[0], count)
        case RootNode(_, position, colors):
            return RootNode(new_children[0], position, colors)
        case SymbolNode(index, parameters):
            new_parameters = list(parameters)
            for i, p in enumerate(new_children):
                new_parameters[i] = p
            return SymbolNode(index, tuple(new_parameters))
        case _:
            return knode  # Leaf nodes remain unchanged


# Traversal


def children(knode: KNode) -> Iterator[KNode]:
    """Unified API to access children of standard KNodes nodes"""
    match knode:
        case CollectionNode(children):
            return iter(children)
        case RepeatNode(node, count):
            children = [node]
            if isinstance(count, VariableNode):
                children.append(count)
            return iter(children)
        case RootNode(node, position, colors):
            children = [node]
            if isinstance(position, VariableNode):
                children.append(position)
            if isinstance(colors, VariableNode):
                children.append(colors)
            return iter(children)
        case SymbolNode(_, parameters):
            return iter(
                (param for param in parameters if isinstance(param, KNode))
            )
        case _:
            return iter(())


def get_subvalues(obj: BitLengthAware) -> Iterator[BitLengthAware]:
    """
    Yields all BitLengthAware subvalues of a BitLengthAware object, assuming it's a dataclass.
    For tuple or list fields, yields each BitLengthAware element.

    Args:
        obj: A BitLengthAware object (e.g., KNode, Primitive, MoveValue).

    Yields:
        BitLengthAware: Subvalues that are instances of BitLengthAware.
    """
    return dataclass_subvalues(obj)


# Traversal functions


# bitlength aware
def breadth_first_preorder_bitlengthaware(
    root: BitLengthAware,
) -> Iterator[BitLengthAware]:
    return breadth_first_preorder(get_subvalues, root)


def depth_first_preorder_bitlengthaware(
    root: BitLengthAware,
) -> Iterator[BitLengthAware]:
    return depth_first_preorder(get_subvalues, root)


# Knodes
def breadth_first_preorder_knode(node: KNode[T] | None) -> Iterator[KNode[T]]:
    return depth_first_preorder(children, node)


def depth(node: KNode) -> int:
    """Returns the depth of a Kolmogorov tree"""
    max_depth = 0
    layer = (node,)
    while layer:
        max_depth += 1
        layer = next_layer(layer)

    return max_depth


# Basic operations on Nodes
def reverse_node(knode: KNode[T]) -> KNode[T]:
    """
    Reverses the structure of the given KNode based on its type.
    Assume that SumNode has already been run-lenght-encoded

    Args:
        knode: The KNode to reverse.

    Returns:
        KNode: The reversed node.
    """
    match knode:
        case ProductNode(children):
            reversed_children = tuple(
                reverse_node(child) for child in reversed(children)
            )
            return type(knode)(reversed_children)
        case SumNode(children):
            reversed_children = frozenset(
                reverse_node(child) for child in children
            )
            return type(knode)(reversed_children)
        case RepeatNode(node, count):
            return RepeatNode(reverse_node(node), count)
        case SymbolNode(index, parameters):
            reversed_params = tuple(
                reverse_node(p) if isinstance(p, KNode) else p
                for p in reversed(parameters)
            )
            return SymbolNode(index, reversed_params)
        case RootNode(node, position, colors):
            nnode = reverse_node(node)
            return RootNode(nnode, position, colors)
        case _:
            return knode


def shift(node: KNode[T], k: int) -> KNode[T]:
    """
    Shifts the values of all PrimitiveNodes in the KolmogorovTree that contain Alphabet subclasses by k.

    This function recursively traverses the KolmogorovTree using the postmap function and applies the shift
    operation to any PrimitiveNode that holds a value of a type that is a subclass of Alphabet. The shift
    operation is defined by the `shift` method of the Alphabet subclass.

    Parameters:
    -----------
    node : KNode[T]
        The root node of the KolmogorovTree to shift.
    k : int
        The amount to shift the values by.

    Returns:
    --------
    KNode[T]
        A new KolmogorovTree with the same structure, but with all shiftable values shifted by k.

    Examples:
    ---------
    >>> node = PrimitiveNode(MoveValue(1))
    >>> shifted = shift(node, 1)
    >>> shifted.data  # Assuming MoveValue shifts modulo 8
    2
    >>> node = ProductNode((PrimitiveNode(MoveValue(1)), PrimitiveNode(CountValue(2))))
    >>> shifted = shift(node, 1)
    >>> shifted.children[0].data  # Shifted MoveValue
    2
    >>> shifted.children[1].data  # Unchanged CountValue
    2
    """
    return postmap(node, lambda n: shift_f(n, k))


def encode_run_length(primitives: Iterable[PrimitiveNode[T]]) -> ProductNode[T]:
    """
    Encodes an iterable of PrimitiveNode[T] into a ProductNode[T] using run-length encoding.
    Consecutive identical PrimitiveNodes are compressed into RepeatNodes when the sequence
    length is 3 or greater.

    Args:
        primitives: An iterable of PrimitiveNode[T] objects.

    Returns:
        A ProductNode[T] containing a tuple of KNode[T] (PrimitiveNode[T] or RepeatNode[T]).
    """
    # List to store the sequence of nodes
    sequence: list[KNode[T]] = []

    # Convert the iterable to an iterator and handle the empty case
    iterator = iter(primitives)
    try:
        current = next(iterator)  # Get the first node
    except StopIteration:
        return ProductNode(
            tuple()
        )  # Return an empty ProductNode if iterable is empty

    # Initialize count for the current sequence
    count = 1

    # Process each node in the iterable
    for node in iterator:
        if node == current:
            # If the node matches the current one, increment the count
            count += 1
        else:
            # If the node differs, decide how to encode the previous sequence
            if count >= 3:
                # For sequences of 3 or more, use a RepeatNode
                repeat_node = RepeatNode(current, CountValue(count))
                sequence.append(repeat_node)
            else:
                # For sequences of 1 or 2, append individual PrimitiveNodes
                sequence.extend([current] * count)
            # Update current node and reset count
            current = node
            count = 1

    # Handle the last sequence after the loop
    if count >= 3:
        repeat_node = RepeatNode(current, CountValue(count))
        sequence.append(repeat_node)
    else:
        sequence.extend([current] * count)

    # Return the sequence as a ProductNode
    return ProductNode(tuple(sequence))


# Tests
def is_symbolized(node: KNode) -> bool:
    """Return True if and only if node contains at least one SymbolNode in its subnodes"""
    subnodes = breadth_first_preorder_knode(node)
    return any(isinstance(node, SymbolNode) for node in subnodes)


def is_abstraction(node: KNode) -> bool:
    """Return True if and only if node contains at least one VariableNode in its subvalues"""
    # For now, all VariableNodes are direct children of the node
    # If it cease to be the case one day, replace by
    # sub_values = breadth_first_preorder_knode(node)
    # any(isinstance(value, VariableValue) for value in sub_values) # VariableNode -> VariableValue, so both one or the other works
    sub_values = get_subvalues(node)
    return any(
        isinstance(value, VariableNode) or isinstance(value, VariableValue)
        for value in sub_values
    )


# Retrievial
def contained_symbols(knode: KNode) -> tuple[IndexValue, ...]:
    subnodes = breadth_first_preorder_knode(knode)
    return tuple(
        node.index for node in subnodes if isinstance(node, SymbolNode)
    )


def arity(node: KNode) -> int:
    """Return the max index of the node variable, which is the arity of an abstraction"""
    # For now, all VariableNodes are direct children of the node
    # If it cease to be the case one day, replace by
    # sub_values = breadth_first_preorder_knode(node)
    subvalues = get_subvalues(node)
    variable_numbers = [
        value.value for value in subvalues if isinstance(value, VariableValue)
    ]
    if variable_numbers:
        return max(variable_numbers)
    return 0


# Compression
def find_repeating_pattern(
    nodes: Sequence[KNode[T]], offset: int
) -> tuple[KNode[T] | None, int, bool]:
    """
    Finds the best repeating pattern in a list of KNodes starting at the given offset, including alternating (reversed) patterns,
    optimizing for bit-length compression.

    Args:
        nodes: List of KNode[T] to search for patterns.
        offset: Starting index in the list to begin the search.

    Returns:
        Tuple of (pattern_node, count, is_reversed):
        - pattern_node: The KNode representing the repeating unit (None if no pattern found).
        - count: Number of repetitions (positive or negative based on reversal).
        - is_reversed: True if the pattern alternates with its reverse.

    """
    if offset >= len(nodes):
        return None, 0, False

    best_pattern: KNode[T] | None = None
    best_count = 0
    best_bit_gain = 0
    best_reverse = False
    max_pattern_len = (
        len(nodes) - offset + 1
    ) // 2  # Need at least 2 occurrences

    for pattern_len in range(1, max_pattern_len + 1):
        pattern = nodes[offset : offset + pattern_len]
        pattern_node = (
            construct_product_node(pattern) if len(pattern) > 1 else pattern[0]
        )

        for reverse in [False, True]:
            count = 1
            i = offset + pattern_len
            while i < len(nodes):
                if i + pattern_len > len(nodes):
                    break

                match = True
                segment = nodes[i : i + pattern_len]
                compare_node = (
                    reverse_node(pattern_node)
                    if (reverse and count % 2 == 1)
                    else pattern_node
                )

                # Compare the segment with the pattern or its reverse
                if len(segment) == pattern_len:
                    segment_node = (
                        construct_product_node(segment)
                        if len(segment) > 1
                        else segment[0]
                    )
                    if segment_node != compare_node:
                        match = False
                else:
                    match = False

                if match:
                    count += 1
                    i += pattern_len
                else:
                    break

            if count > 1:
                # Calculate bit-length savings
                original_bits = sum(
                    node.bit_length()
                    for node in nodes[offset : offset + pattern_len * count]
                )
                compressed = RepeatNode(
                    pattern_node, CountValue(count if not reverse else -count)
                )
                compressed_bits = compressed.bit_length()
                bit_gain = original_bits - compressed_bits

                if bit_gain > best_bit_gain:
                    best_pattern = pattern_node
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return (
        best_pattern,
        best_count if not best_reverse else -best_count,
        best_reverse,
    )


def flatten_sum(node: SumNode[T]) -> KNode[T]:
    # If there is a single node, which is not a RepeatNode, flatten the tuple
    if len(node.children) > 1:
        return node

    child = next(iter(node.children))
    if isinstance(child, RepeatNode):
        return node

    return child


def flatten_product(node: ProductNode[T]) -> KNode[T]:
    if len(node.children) == 0:
        raise ValueError(f"A product node contains no children: {node}")

    if len(node.children) > 1:
        return node

    return node.children[0]


def factorize_tuple(node: KNode[T]) -> KNode[T]:
    """
    Compresses a ProductNode or SumNode by detecting and encoding repeating patterns with RepeatNodes.

    Args:
        node: A KNode[T], typically a ProductNode or SumNode, to factorize.

    Returns:
        A new KNode[T] with repeating patterns compressed.
    """
    if isinstance(node, SumNode):
        # For SumNode, delegate to get_iterator for arithmetic sequence compression
        children = get_iterator(node.children)
        # If there is a no repeat single node, flatten the tuple
        # For the iterator hack, no unpacking
        return flatten_sum(SumNode(children))

    if not isinstance(node, ProductNode):
        return node

    # Handle ProductNode compression
    children = list(node.children)
    if len(children) < 2:
        return node

    simplified: list[KNode[T]] = []
    i = 0
    while i < len(children):
        pattern, count, is_reversed = find_repeating_pattern(children, i)
        if pattern is not None and abs(count) > 1:
            # Calculate bit lengths
            repeat_count = abs(count)
            original_bits = sum(
                child.bit_length()
                for child in children[
                    i : i
                    + repeat_count
                    * (
                        len(pattern.children)
                        if isinstance(pattern, ProductNode)
                        else 1
                    )
                ]
            )
            repeat_node = RepeatNode(pattern, CountValue(count))
            compressed_bits = repeat_node.bit_length()

            # Only use RepeatNode if it reduces bit length
            if compressed_bits < original_bits:
                simplified.append(repeat_node)
                i += repeat_count * (
                    len(pattern.children)
                    if isinstance(pattern, ProductNode)
                    else 1
                )
            else:
                # If no bit savings, append the original nodes one by one
                num_nodes = (
                    len(pattern.children)
                    if isinstance(pattern, ProductNode)
                    else 1
                )
                simplified.extend(children[i : i + num_nodes])
                i += num_nodes

        else:
            simplified.append(children[i])
            i += 1

    # Reconstruct the ProductNode with simplified children
    if len(simplified) == 1 and simplified[0] == node:
        return node  # Avoid unnecessary reconstruction
    return construct_product_node(simplified)


# High-order functions


def postmap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode[T]:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.
        factorize: Wether to automatically factorize the TupleNodes of the tree

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    return postorder_map(
        knode,
        f,
        children,
        lambda node, kids: reconstruct_knode(node, kids, factorize),
    )


def premap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode:
    """
    Map a function alongside a KNode. It updates the base node first, then updates the children

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.
        factorize: Wether to automatically factorize the TupleNodes of the tree

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    return preorder_map(
        knode,
        f,
        children,
        lambda node, kids: reconstruct_knode(node, kids, factorize),
    )


def postmap_unsafe(
    knode: KNode, f: Callable[[KNode], KNode | None]
) -> KNode | None:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    match knode:
        case CollectionNode(children):
            mapped_children = tuple(
                node
                for child in children
                if (node := postmap_unsafe(child, f)) is not None
            )
            return f(factorize_tuple(type(knode)(mapped_children)))
        case RepeatNode(node, count):
            nnode = postmap_unsafe(node, f)
            return f(RepeatNode(nnode, count)) if nnode is not None else None
        case RootNode(node, position, color):
            nnode = postmap_unsafe(node, f)
            return (
                f(RootNode(nnode, position, color))
                if nnode is not None
                else None
            )
        case SymbolNode(index, parameters):
            nparameters = tuple(
                nparam
                for p in parameters
                if (
                    nparam := (
                        postmap_unsafe(p, f) if isinstance(p, KNode) else p
                    )
                )
                is not None
            )
            return f(SymbolNode(index, nparameters))
        case _:
            return f(knode)


# Decompression
def expand_repeats(
    node: SumNode[T] | ProductNode[T] | RepeatNode[T] | PrimitiveNode[T],
) -> SumNode[T] | ProductNode[T] | PrimitiveNode[T]:
    """Used to uncompress knodes"""

    def expand_repeat_f(knode: KNode) -> KNode:
        """Expand repeats"""
        if isinstance(knode, RepeatNode):
            if isinstance(knode.count, VariableNode):
                raise TypeError("Trying to uncompress a variable repeat")
            return ProductNode(tuple([knode.node] * knode.count.value))
        else:
            return knode

    expanded = postmap(node, expand_repeat_f, factorize=False)
    return cast(SumNode[T] | ProductNode[T] | PrimitiveNode[T], expanded)


# Symbol resolution and pattern finding

Bindings = dict[int, BitLengthAware]


def matches(pattern: KNode[T], subtree: KNode[T]) -> Bindings | None:
    """
    Determines if a pattern matches a subtree, returning variable bindings if successful.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree to match against.

    Returns:
        Bindings | None: A dictionary of variable bindings if the match succeeds, or None if it fails.
    """
    if not isinstance(pattern, KNode) or not isinstance(subtree, KNode):
        raise TypeError("Both pattern and subtree must be KNode instances.")
    bindings: Bindings = {}
    if unify(pattern, subtree, bindings):
        return bindings
    return None


def unify(
    pattern: BitLengthAware, subtree: BitLengthAware, bindings: Bindings
) -> bool:
    """
    Recursively unifies a pattern with a subtree, updating bindings in place.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree or primitive to match against.
        bindings: Current bindings of variables to subtrees or primitives, modified in place.

    Returns:
        bool: True if unification succeeds, False if it fails.
    """
    if isinstance(pattern, VariableNode):
        index = pattern.index.value
        if index in bindings:
            return bindings[index] == subtree
        bindings[index] = subtree
        return True

    if not isinstance(subtree, type(pattern)):
        return False

    match pattern:
        case Primitive(value):
            return isinstance(subtree, Primitive) and value == subtree.value
        case CollectionNode(children):
            subtree = cast(CollectionNode, subtree)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            for p_child, s_child in zip(children, s_children):
                if not unify(p_child, s_child, bindings):
                    return False
            return True
        case RepeatNode(node, count):
            subtree = cast(RepeatNode, subtree)
            s_node = subtree.node
            s_count = subtree.count
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != s_count:
                        return False
                else:
                    bindings[index] = s_count
            elif count != s_count:
                return False
            return unify(node, s_node, bindings)
        case RootNode(node, position, colors):
            subtree = cast(RootNode, subtree)
            if not unify(position, subtree.position, bindings):
                return False
            if not unify(colors, subtree.colors, bindings):
                return False
            return unify(node, subtree.node, bindings)
        case SymbolNode(index, parameters):
            subtree = cast(SymbolNode, subtree)
            if index != subtree.index or len(parameters) != len(
                subtree.parameters
            ):
                return False
            for param, s_param in zip(parameters, subtree.parameters):
                if not unify(param, s_param, bindings):
                    return False
            return True
        case RectNode(height, width):
            subtree = cast(RectNode, subtree)
            if not unify(height, subtree.height, bindings):
                return False
            if not unify(width, subtree.width, bindings):
                return False
            return True
        case _:
            return pattern == subtree


def abstract_node(
    index: IndexValue, pattern: KNode[T], node: KNode[T]
) -> KNode[T]:
    """
    If node matches the pattern, it replaces it by a SymbolNode with the right bindings.
    Else, it returns the node as is.
    """

    if pattern == node:
        return SymbolNode(index, (), pattern.bit_length())

    bindings = matches(pattern, node)

    if bindings is not None:
        # TODO
        # Understrand why 'PrimitiveNode(MoveValue(0))' as a fallback
        params = tuple(
            bindings.get(j, PrimitiveNode(MoveValue(0)))
            for j in range(max(bindings.keys(), default=-1) + 1)
        )
        return SymbolNode(index, params, pattern.bit_length())

    return node


def substitute_variables(abstraction: KNode[T], params: Parameters) -> KNode[T]:
    """
    Helper function to substitute variables, including those that are not traversed
    during a simple node traversal
    """
    match abstraction:
        case VariableNode(index) if index.value < len(params):
            node = params[index.value]
            if not isinstance(node, KNode):
                raise TypeError(
                    f"Trying to substitute a non-node parameter to a variable encountered during node traversal: {node}"
                )
            return node
        case RepeatNode(node, VariableNode(index)) if index.value < len(params):
            count = params[index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return RepeatNode(node, count)
        case RootNode(node, VariableNode(index1), VariableNode(index2)) if (
            index1.value < len(params) and index2.value < len(params)
        ):
            position = params[index1.value]
            colors = params[index2.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, VariableNode(index), colors) if index.value < len(
            params
        ):
            position = params[index.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, position, VariableNode(index)) if index.value < len(
            params
        ):
            colors = params[index.value]
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RectNode(VariableNode(index1), VariableNode(index2)) if (
            index1.value < len(params) and index2.value < len(params)
        ):
            height = params[index1.value]
            width = params[index2.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case RectNode(VariableNode(index), width) if index.value < len(params):
            height = params[index.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            return RectNode(height, width)
        case RectNode(height, VariableNode(index)) if index.value < len(params):
            width = params[index.value]
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case _:
            return abstraction


def reduce_abstraction(abstraction: KNode[T], params: Parameters) -> KNode[T]:
    """
    Substitutes variable placeholders in the template with the corresponding parameters.

    This function recursively traverses the template node and replaces VariableNodes
    with the provided parameters. It also handles composite nodes by substituting variables
    in their children or attributes.

    Args:
        abstraction (KNode): The abstract node to process.
        params (tuple[Any, ...]): The parameters to substitute for variables.

    Returns:
        KNode: The abstraction with variables replaced by parameters.
    """
    return premap(abstraction, lambda node: substitute_variables(node, params))


@functools.cache
def resolve_symbols(knode: KNode[T], symbols: tuple[KNode[T], ...]) -> KNode[T]:
    """
    Resolves symbolic references recursively using the symbol table.

    This function traverses the Kolmogorov Tree and replaces SymbolNodes with their
    definitions from the symbol table, handling parameter substitution. It recursively
    resolves all sub-nodes for composite node types.

    Args:
        node (KNode): The node to resolve.
        symbols (list[KNode]): The list of symbol definitions.

    Returns:
        KNode: The resolved node with symbols expanded.
    """

    def resolve_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and 0 <= node.index.value < len(
            symbols
        ):
            return reduce_abstraction(
                symbols[node.index.value], node.parameters
            )
        return node

    return premap(knode, resolve_f, factorize=False)


# Factory functions for common patterns
def create_move_node(direction: int) -> PrimitiveNode:
    """Creates a node for a directional move."""
    return PrimitiveNode(MoveValue(direction))


def create_moves_sequence(directions: str) -> KNode | None:
    """Creates a sequence of moves from a direction string."""
    if not directions:
        return None
    moves = [create_move_node(int(d)) for d in directions]
    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len
                if (
                    all(
                        moves[i * pattern_len : (i + 1) * pattern_len]
                        == pattern
                        for i in range(repeat_count)
                    )
                    and repeat_count > 1
                ):
                    return RepeatNode(
                        ProductNode(tuple(pattern)), CountValue(repeat_count)
                    )
    return ProductNode(tuple(moves))


def create_rect(height: int, width: int) -> KNode | None:
    """Creates a node for a rectangle shape."""
    if height < 2 or width < 2:
        return None
    first_row = "2" * (width - 1)  # Move right
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"  # Alternate left/right
        other_rows += "3" + direction * (width - 1)  # Down then horizontal
    return create_moves_sequence(first_row + other_rows)


def find_symbol_candidates(
    trees: tuple[KNode[T], ...],
    min_occurrences: int = 2,
    max_patterns: int = 10,
) -> list[KNode]:
    """
    Identifies frequent concrete and abstracted subtrees across multiple KolmogorovTrees.
    Returns the top patterns ranked by bit-length savings.
    """
    # Step 1: Collect all subtrees
    all_subtrees = []
    for tree in trees:
        all_subtrees.extend(breadth_first_preorder_knode(tree))

    # Step 2: Count frequencies and track matches
    pattern_counter = Counter()
    pattern_matches = defaultdict(
        set
    )  # Maps pattern to set of matched subtrees

    for subtree in all_subtrees:
        pattern_counter[subtree] += 1
        pattern_matches[subtree].add(subtree)
        for abs_pattern, params in generate_abstractions(subtree):
            pattern_counter[abs_pattern] += 1
            pattern_matches[abs_pattern].add(subtree)

    # Step 3: Filter patterns
    common_patterns = []
    seen_patterns = set()

    for pattern, count in pattern_counter.items():
        if count < min_occurrences or pattern in seen_patterns:
            continue
        if any(
            isinstance(n, VariableNode)
            for n in breadth_first_preorder_knode(pattern)
        ):
            if len(pattern_matches[pattern]) >= min_occurrences:
                common_patterns.append(pattern)
                seen_patterns.add(pattern)
        else:
            common_patterns.append(pattern)
            seen_patterns.add(pattern)

    # Step 4: Calculate bit gain and filter for positive savings
    def bit_gain(pat: KNode) -> float:
        count = pattern_counter[pat]
        # avg_len = sum(s.bit_length() for s in pattern_matches[pat]) / count
        current_len = sum(s.bit_length() for s in pattern_matches[pat])
        param_len = (
            sum(
                p.bit_length()
                for s in pattern_matches[pat]
                for _, ps in generate_abstractions(s)
                if pat == _
                for p in ps
            )
            / count
        )
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        # return (count - 1) * (avg_len - symb_len) - pat.bit_length()
        return current_len - (count * symb_len + pat.bit_length())

    # Might need
    # and pattern_counter[node] > tree_count (lattice_count)
    # You may want more than 1 per full tree ?
    common_patterns = [pat for pat in common_patterns if bit_gain(pat) > 0]

    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]


def symbolize_pattern(
    trees: tuple[KNode[T], ...],
    symbols: tuple[KNode[T], ...],
    new_symbol: KNode[T],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    index = IndexValue(len(symbols))
    trees = tuple(abstract_node(index, new_symbol, tree) for tree in trees)
    symbols = tuple(
        abstract_node(index, new_symbol, tree) for tree in symbols
    ) + (new_symbol,)
    return trees, symbols


def greedy_symbolization(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Symbolize trees by replacing the common pattern with SymbolNodes.
    It only do the most common pattern, because symbolizing one pattern potentially changes the bit length saving of the rest
    """

    # While there is a pattern whose abstraction leads to bit gain savings
    common_patterns = find_symbol_candidates(trees + symbols)
    while common_patterns:
        # Abstract the best one
        new_symbol = common_patterns[0]
        trees, symbols = symbolize_pattern(trees, symbols, new_symbol)
        common_patterns = find_symbol_candidates(trees + symbols)

    return (trees, symbols)


def symbolize(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    # Phase 1: Non-symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not is_symbolized(c) and not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # # Phase 2: Symbolic patterns by depth
    # for depth in range(1, max_depth(trees) + 1):
    #     while True:
    #         candidates = [c for c in find_symbol_candidates(trees + symbols) if len(contained_symbols(c)) <= depth]
    #         if not candidates:
    #             break
    #         trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 2: Including Symbolic patterns:
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 3: Include roots
    while True:
        candidates = find_symbol_candidates(
            trees + symbols
        )  # All candidates, including RootNodes
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    return trees, symbols


## Re-factorization:


def factor_by_existing_symbols(
    tree: KNode[T], symbols: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Factors a tree against an existing symbol table, replacing matches with SymbolNodes.

    Args:
        tree: The tree to factor.
        symbols: The symbol table to check against.

    Returns:
        Factored tree with applicable SymbolNode replacements.
    """

    def factor_node(node: KNode[T]) -> KNode[T]:
        for i, symbol in enumerate(symbols):
            abstracted = abstract_node(IndexValue(i), symbol, node)
            if abstracted != node:  # If abstraction occurred
                return abstracted
        return node

    return postmap(tree, factor_node, factorize=True)


def remap_symbol_indices(
    tree: KNode[T], mapping: list[int], tree_idx: int
) -> KNode[T]:
    """
    Updates SymbolNode indices in a tree based on the provided mapping.
    Used to remap the tree elements.

    Args:
        tree: The tree to update.
        mapping: List mapping old indices to new ones.
        tree_idx: Index of the tree for debugging purposes.

    Returns:
        Updated tree with remapped SymbolNode indices.
    """

    def update_node(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters, node.reference_length)
        return node

    return postmap(tree, update_node, factorize=False)


def remap_sub_symbols(
    symbol: KNode[T], mapping: list[int], original_table: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Remaps SymbolNode indices within a symbol based on the provided mapping.
    Used to remap the symbol tables elements.

    Args:
        symbol: The symbol to update.
        mapping: List mapping old indices to new ones.
        original_table: The original symbol table for resolution if needed.

    Returns:
        Updated symbol with remapped SymbolNode indices.
    """

    def update_index(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters, node.reference_length)
        return node

    return postmap(symbol, update_index, factorize=False)


def merge_symbol_tables(
    symbol_tables: Sequence[tuple[KNode[T], ...]],
) -> tuple[tuple[KNode[T], ...], list[list[int]]]:
    """
    Merges multiple symbol tables into a unified table, returning the table and mappings.
    It choose symbols so to minimize the total bit length.
    Note: Clean resymbolisation is often preferable.

    Args:
        symbol_tables: List of symbol tables to merge.

    Returns:
        Tuple of the unified symbol table and a list of mappings (old index -> new index) for each input table.
    """
    unified_symbols: list[KNode] = []
    mappings: list[list[int]] = [[] for _ in range(len(symbol_tables))]

    equivalence_classes: defaultdict[KNode, list[tuple[KNode, int]]] = (
        defaultdict(list)
    )  # Store symbol classes, in an union-find like structure. Each classes contains the symbols and their tables
    dependency_graph: defaultdict[tuple[KNode, int], set[int]] = defaultdict(
        set
    )  # Store for each symbol the symbols it depends on

    # Step 1: Collect all resolved symbolsa and build the dependency graph
    for i, table in enumerate(symbol_tables):
        for symbol in table:
            resolved_symbol = resolve_symbols(symbol, table)
            equivalence_classes[resolved_symbol].append((symbol, i))
            # Track dependencies: which symbols this symbol references in its table
            subsymbols = set(index.value for index in contained_symbols(symbol))
            dependency_graph[(symbol, i)] |= subsymbols

    # Step 2: Select optimal symbols per equivalence class
    selected_symbols = {}
    for resolved, symbols_in_class in equivalence_classes.items():
        # Prefer abstracted symbols (those with variables)
        abstracted = [s for s, _ in symbols_in_class if is_abstraction(s)]
        if abstracted:
            # Select the one with the most variables, then smallest bit_length
            selected = max(
                abstracted, key=lambda s: (arity(s), -s.bit_length())
            )
        else:
            # Select the one with the smallest bit_length
            selected = min(
                (s for s, _ in symbols_in_class), key=lambda s: s.bit_length()
            )
        selected_symbols[resolved] = selected

    # Step 3: Update dependency graph with selected symbols
    new_dependency_graph = defaultdict(set)
    symbol_to_selected = {}
    for resolved, selected in selected_symbols.items():
        for symbol, table_idx in equivalence_classes[resolved]:
            symbol_to_selected[(symbol, table_idx)] = selected
            table = symbol_tables[table_idx]
            for dep_idx in dependency_graph[(symbol, table_idx)]:
                if dep_idx < len(table):
                    dep_symbol = table[dep_idx]
                    dep_resolved = resolve_symbols(dep_symbol, table)
                    dep_selected = selected_symbols[dep_resolved]
                    new_dependency_graph[selected].add(dep_selected)

    # Step 4: Topological sort to order symbols in the unified table
    in_degree = {s: len(deps) for s, deps in new_dependency_graph.items()}
    for s in selected_symbols.values():
        if s not in in_degree:
            in_degree[s] = 0
    queue = deque([s for s, deg in in_degree.items() if deg == 0])
    unified_symbols = []

    while queue:
        symbol = queue.popleft()
        unified_symbols.append(symbol)
        for dependent in new_dependency_graph:
            if symbol in new_dependency_graph[dependent]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    symbol_to_index = {s: i for i, s in enumerate(unified_symbols)}
    mappings = [[] for _ in symbol_tables]

    for table_idx, table in enumerate(symbol_tables):
        for symbol in table:
            resolved = resolve_symbols(symbol, table)
            selected = selected_symbols[resolved]
            mappings[table_idx].append(symbol_to_index[selected])
    # TODO

    return tuple(unified_symbols), mappings


def symbolize_together(
    trees: tuple[KNode[T], ...], symbol_tables: Sequence[tuple[KNode[T], ...]]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Merges independently symbolized trees into a unified symbol table and re-symbolizes them together.

    Args:
        trees: Tuple of KNode[MoveValue] trees, each potentially containing SymbolNodes.
        symbol_tables: List of symbol tables corresponding to each tree (or empty if unsymbolized).

    Returns:
        Tuple of updated trees and the unified symbol table.
    """
    if symbol_tables:
        if len(symbol_tables) != len(trees):
            raise ValueError(
                f"There are only {len(symbol_tables)} symbol tables for {len(trees)} trees"
            )
        # Step 1: Merge symbol tables into a unified table with index remapping
        unified_symbols, mappings = merge_symbol_tables(symbol_tables)

        # Step 2: Update trees with the new symbol indices
        updated_trees = tuple(
            remap_symbol_indices(tree, mapping, i)
            for i, (tree, mapping) in enumerate(zip(trees, mappings))
        )

        # Step 3: Factor existing patterns against the unified symbol table
        factored_trees = tuple(
            factor_by_existing_symbols(tree, unified_symbols)
            for tree in updated_trees
        )
    else:
        factored_trees = trees
        unified_symbols = tuple()

    # Step 4: Find and symbolize new common patterns across all trees
    final_trees, final_symbols = symbolize(factored_trees, unified_symbols)

    return final_trees, final_symbols


# Distance
def bla_distance(
    base: BitLengthAware | None,
    target: BitLengthAware | None,
    symbols: tuple[KNode[T], ...],
) -> tuple[int, tuple[Operation[BitLengthAware], ...]]:
    """
    Computes the distance between two BitLengthAware objects, handling KNodes recursively via k_divergence, and tracks transformations.
    None is considered as a neutral element used to complete the BitLengthAware class.

    Args:
        base: First BitLengthAware object (or None), acts like the base element for the transformation..
        target: Second BitLengthAware object (or None), acts like the target node for the transformation.
        symbols: Symbol table for resolving SymbolNodes.

    Returns:
        int: Distance between base and target.
        transformations:
    """

    distance = 0
    transformations: list[Operation[BitLengthAware]] = []

    match base, target:
        case None, None:
            pass  # May be Identity(None) ? to discuss as it changes the return type only for this case
        case _, None:
            distance = base.bit_length()
            transformations.append(Delete(base))
        case None, _:
            distance = target.bit_length()
            transformations.append(Add(target))
        case Primitive(base_value), Primitive(target_value):
            pass
        case _, _:
            pass

    return distance, tuple(transformations)


# Tests
def test_encode_run_length():
    # Test Case 1: Empty Input
    result = encode_run_length([])
    assert result == ProductNode(()), (
        "Test Case 1 Failed: Empty input should return empty ProductNode"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = encode_run_length([node])
    assert result == ProductNode((node,)), (
        "Test Case 2 Failed: Single node should be wrapped in ProductNode"
    )

    # Test Case 3: No Repeats
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 3 Failed: No repeats should remain uncompressed"
    )

    # Test Case 4: Short Repeats
    nodes = [
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
    ]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 4 Failed: Repeats less than 3 should not be compressed"
    )

    # Test Case 5: Long Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = encode_run_length(nodes)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),)
    )
    assert result == expected, (
        "Test Case 5 Failed: Three identical nodes should be compressed"
    )

    # Test Case 6: Mixed Sequences
    nodes = (
        [PrimitiveNode(MoveValue(1))] * 3
        + [PrimitiveNode(MoveValue(2))]
        + [PrimitiveNode(MoveValue(3))] * 2
    )
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    assert result == expected, (
        "Test Case 6 Failed: Mixed sequence compression incorrect"
    )

    # Test Case 7: Multiple Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [
        PrimitiveNode(MoveValue(2))
    ] * 4
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 7 Failed: Multiple repeat sequences not handled correctly"
    )

    # Test Case 8: All Identical
    nodes = [PrimitiveNode(MoveValue(5))] * 10
    result = encode_run_length(nodes)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(5)), CountValue(10)),)
    )
    assert result == expected, (
        "Test Case 8 Failed: All identical nodes should compress into one RepeatNode"
    )

    # Test Case 9: Input as Iterator
    nodes = iter([PrimitiveNode(MoveValue(1))] * 4)
    result = encode_run_length(nodes)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(4)),)
    )
    assert result == expected, (
        "Test Case 9 Failed: Iterator input not handled correctly"
    )

    # Test Case 10: Alternating Nodes
    nodes = [
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
    ]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 11 Failed: Alternating nodes should not be compressed"
    )

    # Test Case 11: Repeats in Different Positions
    nodes = (
        [PrimitiveNode(MoveValue(1))] * 3
        + [PrimitiveNode(MoveValue(2))]
        + [PrimitiveNode(MoveValue(3))] * 3
        + [PrimitiveNode(MoveValue(4))] * 2
    )
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
            RepeatNode(PrimitiveNode(MoveValue(3)), CountValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 12 Failed: Repeats in different positions not handled correctly"
    )

    print("Test 5: `encode_run_length` tests - Passed")


def test_construct_product_node():
    # Test 1: Empty Input
    # Verifies that an empty iterable returns an empty ProductNode
    result = construct_product_node([])
    expected = ProductNode(())
    assert result == expected, (
        "Test 1 Failed: Empty input should return empty ProductNode"
    )

    # Test 2: Single Node
    # Checks that a single node is wrapped in a ProductNode without changes
    node = PrimitiveNode(MoveValue(1))
    result = construct_product_node([node])
    expected = ProductNode((node,))
    assert result == expected, (
        "Test 2 Failed: Single node should be wrapped in ProductNode"
    )

    # Test 3: Merging Adjacent PrimitiveNodes
    # Ensures consecutive identical PrimitiveNodes are compressed into a RepeatNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = construct_product_node(nodes)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),)
    )
    assert result == expected, (
        "Test 3 Failed: Adjacent PrimitiveNodes should merge into RepeatNode"
    )

    # Test 4: Preserving SumNodes
    # Confirms that SumNodes are kept intact and not merged
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(3)), PrimitiveNode(MoveValue(4))])
    )
    nodes = [PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2))]
    result = construct_product_node(nodes)
    expected = ProductNode(
        (PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2)))
    )
    assert result == expected, "Test 4 Failed: SumNodes should be preserved"
    print("Test 6 - Product Node consstructor basic tests passed successfully!")


def test_shift():
    """
    Tests the shift operation across different node types in the KolmogorovTree.
    The shift operation should modify MoveValue primitives by adding k modulo 8,
    while leaving non-MoveValue primitives unchanged.
    """
    # Test 1: Shifting a PrimitiveNode with MoveValue
    node = create_move_node(2)
    shifted = shift(node, 1)
    assert isinstance(shifted, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted.data == 3, "MoveValue should shift from 2 to 3 with k=1"

    # Test 2: Shifting by 0 (no change)
    shifted_zero = shift(node, 0)
    assert shifted_zero == node, "Shifting by 0 should return the same node"
    assert isinstance(shifted_zero, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_zero.data == 2, "Value should remain 2 when k=0"

    # Test 3: Shifting SumNode
    sum_node = SumNode(frozenset([create_move_node(0), create_move_node(4)]))
    shifted_sum = shift(sum_node, 2)
    assert isinstance(shifted_sum, SumNode), (
        "Shifted result should be a SumNode"
    )
    assert len(shifted_sum.children) == 2, "SumNode should retain 2 children"
    primitive_children = ensure_all_instances(
        shifted_sum.children, PrimitiveNode
    )

    data_values = {child.data for child in primitive_children}
    assert data_values == {2, 6}, "Children should have data values 2 and 6"

    # Test 4: Shifting RepeatNode
    sequence = ProductNode((create_move_node(2), create_move_node(3)))
    repeat = RepeatNode(sequence, CountValue(3))
    shifted_repeat = shift(repeat, 1)
    assert isinstance(shifted_repeat, RepeatNode), (
        "Shifted result should be a RepeatNode"
    )
    assert isinstance(shifted_repeat.node, ProductNode), (
        "Repeated node should be a ProductNode"
    )
    assert isinstance(shifted_repeat.node.children[0], PrimitiveNode), (
        "First child should be PrimitiveNode"
    )
    assert shifted_repeat.node.children[0].data == 3, (
        "First MoveValue should shift to 3"
    )
    assert isinstance(shifted_repeat.node.children[1], PrimitiveNode), (
        "Second child should be PrimitiveNode"
    )
    assert shifted_repeat.node.children[1].data == 4, (
        "Second MoveValue should shift to 4"
    )
    assert isinstance(shifted_repeat.count, CountValue), (
        "COunt should be CountValue"
    )
    assert shifted_repeat.count.value == 3, "Count should remain unchanged"

    # Test 5: Shifting SymbolNode
    param1 = create_move_node(1)
    param2 = PaletteValue(frozenset({2}))
    symbol = SymbolNode(IndexValue(0), (param1, param2))
    shifted_symbol = shift(symbol, 1)
    assert isinstance(shifted_symbol, SymbolNode), (
        "Shifted result should be a SymbolNode"
    )
    assert len(shifted_symbol.parameters) == 2, (
        "SymbolNode should retain 2 parameters"
    )
    assert isinstance(shifted_symbol.parameters[0], PrimitiveNode), (
        "Parameter should be PrimitiveNode"
    )
    assert shifted_symbol.parameters[0].data == 2, (
        "MoveValue parameter should shift to 2"
    )
    assert shifted_symbol.parameters[1] is param2, (
        "Non-shiftable parameter should be unchanged"
    )

    # Test 6: Shifting RootNode
    program = ProductNode((create_move_node(0), create_move_node(1)))
    root = RootNode(program, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    shifted_root = shift(root, 2)
    assert isinstance(shifted_root, RootNode), (
        "Shifted result should be a RootNode"
    )
    assert isinstance(shifted_root.node, ProductNode), (
        "Root program should be a ProductNode"
    )
    assert isinstance(shifted_root.node.children[0], PrimitiveNode), (
        "First child should be PrimitiveNode"
    )
    assert shifted_root.node.children[0].data == 2, (
        "First MoveValue should shift to 2"
    )
    assert isinstance(shifted_root.node.children[1], PrimitiveNode), (
        "Second child should be PrimitiveNode"
    )
    assert shifted_root.node.children[1].data == 3, (
        "Second MoveValue should shift to 3"
    )
    assert shifted_root.position == CoordValue((0, 0)), (
        "Position should be unchanged"
    )
    assert shifted_root.colors == PaletteValue(frozenset({1})), (
        "Colors should be unchanged"
    )

    # Test 7: Shifting with large k (wrapping around)
    node_large = create_move_node(7)
    shifted_large = shift(node_large, 10)  # 7 + 10 = 17 ≡ 1 mod 8
    assert isinstance(shifted_large, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_large.data == 1, "Large shift should wrap around to 1"

    # Test 8: Shifting with negative k
    shifted_neg = shift(node_large, -3)  # 7 - 3 = 4
    assert isinstance(shifted_neg, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_neg.data == 4, "Negative shift should result in 4"

    # Test 9: Shifting nested composite nodes
    inner_product = ProductNode((create_move_node(5), create_move_node(6)))
    repeat_inner = RepeatNode(inner_product, CountValue(2))
    primitive_outer = create_move_node(7)
    outer_sum = SumNode(frozenset([repeat_inner, primitive_outer]))
    shifted_outer = shift(outer_sum, 1)

    # Verify the shifted outer node is a SumNode
    assert isinstance(shifted_outer, SumNode), (
        "Shifted outer node should be a SumNode"
    )

    # Extract children from frozenset based on type
    children_list = list(shifted_outer.children)
    repeat_nodes = [
        child for child in children_list if isinstance(child, RepeatNode)
    ]
    primitive_nodes = [
        child for child in children_list if isinstance(child, PrimitiveNode)
    ]

    # Ensure the correct number of each type
    assert len(repeat_nodes) == 1, (
        "There should be one RepeatNode in the SumNode"
    )
    assert len(primitive_nodes) == 1, (
        "There should be one PrimitiveNode in the SumNode"
    )

    # Get the single RepeatNode and PrimitiveNode
    shifted_repeat = repeat_nodes[0]
    shifted_primitive = primitive_nodes[0]

    # Verify RepeatNode properties
    assert isinstance(shifted_repeat, RepeatNode), (
        "Child should be a RepeatNode"
    )
    assert isinstance(shifted_repeat.node, ProductNode), (
        "Repeated node should be ProductNode"
    )
    assert len(shifted_repeat.node.children) == 2, (
        "ProductNode should have two children"
    )

    # Add assertion to narrow child types to PrimitiveNode

    primitive_children = ensure_all_instances(
        shifted_repeat.node.children, PrimitiveNode
    )
    data_values = [child.data for child in primitive_children]
    assert data_values == [6, 7], "Nested MoveValues should shift to 6 and 7"

    # Verify PrimitiveNode properties
    assert isinstance(shifted_primitive, PrimitiveNode), (
        "Child should be a PrimitiveNode"
    )
    assert shifted_primitive.data == 0, (
        "Outer MoveValue should shift from 7 to 0"
    )
    # Test 10: Original node remains unchanged
    original_node = create_move_node(2)
    shifted_node = shift(original_node, 1)
    assert isinstance(original_node, PrimitiveNode), (
        "Original node should be PrimitiveNode"
    )
    assert original_node.data == 2, "Original node value should remain 2"
    assert isinstance(shifted_node, PrimitiveNode), (
        "Shifted node should be PrimitiveNode"
    )
    assert shifted_node.data == 3, "Shifted node value should be 3"

    print("Test shift operations - Passed")


def test_get_iterator():
    """Tests the get_iterator function for detecting and compressing arithmetic sequences."""
    # Test Case 1: Empty Input
    result = get_iterator([])
    assert result == frozenset(), (
        "Test Case 1 Failed: Empty input should return an empty frozenset"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = get_iterator([node])
    assert result == frozenset([node]), (
        "Test Case 2 Failed: Single node should return a frozenset with that node"
    )

    # Test Case 3: Sequence with Increment +1
    nodes_pos = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]]
    result_pos = get_iterator(nodes_pos)
    expected_pos_forward = frozenset((RepeatNode(nodes_pos[0], CountValue(3)),))
    expected_pos_backward = frozenset(
        (RepeatNode(nodes_pos[2], CountValue(-3)),)
    )
    assert result_pos in [expected_pos_forward, expected_pos_backward], (
        "Test Case 3 Failed: Sequence [0,1,2] should be compressed to RepeatNode(0, 3) or RepeatNode(2, -3)"
    )

    # Test Case 4: Sequence with Increment -1
    nodes_neg = [PrimitiveNode(MoveValue(i)) for i in [2, 1, 0]]
    result_neg = get_iterator(nodes_neg)
    expected_neg = frozenset((RepeatNode(nodes_neg[0], CountValue(-3)),))
    assert result_neg == expected_neg, (
        "Test Case 4 Failed: Sequence [2,1,0] should be compressed to RepeatNode(2, -3)"
    )

    # Test Case 5: Non-Sequence
    nodes_non = [PrimitiveNode(MoveValue(i)) for i in [0, 5, 1]]
    result_non = get_iterator(nodes_non)
    assert result_non == frozenset(nodes_non), (
        "Test Case 5 Failed: Non-sequence should return original nodes"
    )

    # Test Case 6: Boundary Conditions (Wrap-around with Increment +1)
    nodes_wrap = [PrimitiveNode(MoveValue(i)) for i in [7, 0, 1]]
    result_wrap = get_iterator(nodes_wrap)
    expected_wrap_forward = frozenset(
        (RepeatNode(nodes_wrap[0], CountValue(3)),)
    )
    expected_wrap_backward = frozenset(
        (RepeatNode(nodes_wrap[2], CountValue(-3)),)
    )
    assert result_wrap in [expected_wrap_forward, expected_wrap_backward], (
        "Test Case 6 Failed: Wrap-around sequence [7,0,1] should be compressed"
    )

    # Test Case 7: Long Sequence with Wrap-around, should be equal to the size of the alphabet
    nodes_long = [
        PrimitiveNode(MoveValue(i % 8)) for i in range(10)
    ]  # [0,1,2,3,4,5,6,7,0,1]
    result_long = get_iterator(nodes_long)

    # Add assertion to narrow the type to RepeatNode
    assert len(result_long) == 1 and isinstance(
        next(iter(result_long)), RepeatNode
    ), "Expected a single children"

    node = next(iter(result_long))
    assert isinstance(node, RepeatNode) and node.count == CountValue(8), (
        "Test Case 7 Failed: Count should be 8"
    )

    # Test Case 8: Partial Sequence
    nodes_partial = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2, 4]]
    result_partial = get_iterator(nodes_partial)
    assert result_partial == frozenset(nodes_partial), (
        "Test Case 8 Failed: Partial sequence [0,1,2,4] should not be compressed"
    )

    # Test Case 9: Different Increment
    nodes_diff_inc = [PrimitiveNode(MoveValue(i)) for i in [0, 2, 4, 6]]
    result_diff_inc = get_iterator(nodes_diff_inc)
    assert result_diff_inc == frozenset(nodes_diff_inc), (
        "Test Case 9 Failed: Sequence with increment +2 should not be compressed"
    )

    print("Test get_iterator - Passed")


def test_find_repeating_pattern():
    """Tests the find_repeating_pattern function for detecting repeating patterns."""
    # Test Case 1: Empty Input
    result = find_repeating_pattern([], 0)
    assert result == (None, 0, False), (
        "Test Case 1 Failed: Empty input should return (None, 0, False)"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = find_repeating_pattern([node], 0)
    assert result == (None, 0, False), (
        "Test Case 2 Failed: Single node should return (None, 0, False)"
    )

    # Test Case 3: Simple Repeat
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert pattern == nodes[0], (
        "Test Case 3 Failed: Pattern should be MoveValue(2)"
    )
    assert count == 3, "Test Case 3 Failed: Count should be 3"
    assert not is_reversed, "Test Case 3 Failed: Should not be reversed"

    # Test Case 4: Alternating Repeat
    nodes = [
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
    ]
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert isinstance(pattern, ProductNode) and str(pattern) == "20", (
        "Test Case 4 Failed: Pattern should be 20"
    )
    assert count == -3, "Test Case 4 Failed: Count should be -3 for alternating"
    assert is_reversed, "Test Case 4 Failed: Should be reversed"

    # Test Case 5: Multi-Node Pattern
    pattern_nodes = [PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3))]
    nodes = pattern_nodes * 3  # 2,3,2,3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert isinstance(pattern, ProductNode) and pattern.children == tuple(
        pattern_nodes
    ), "Test Case 5 Failed: Incorrect pattern"
    assert count == 3, "Test Case 5 Failed: Count should be 3"
    assert not is_reversed, "Test Case 5 Failed: Should not be reversed"

    # Test Case 6: No Repeat
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = find_repeating_pattern(nodes, 0)
    assert result == (None, 0, False), (
        "Test Case 6 Failed: Non-repeating sequence should return (None, 0, False)"
    )

    # Test Case 7: Offset Pattern
    nodes = [PrimitiveNode(MoveValue(1))] + [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 1)
    assert pattern == PrimitiveNode(MoveValue(2)), (
        "Test Case 7 Failed: Pattern should be MoveValue(2)"
    )
    assert count == 3, "Test Case 7 Failed: Count should be 3"
    assert not is_reversed, "Test Case 7 Failed: Should not be reversed"

    print("Test find_repeating_pattern - Passed")


def test_factorize_tuple():
    """Tests the factorize_tuple function for compressing ProductNode and SumNode."""
    # Test Case 1: Empty ProductNode
    node = ProductNode(())
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 1 Failed: Empty ProductNode should remain unchanged"
    )

    # Test Case 2: Single Node ProductNode
    node = ProductNode((PrimitiveNode(MoveValue(1)),))
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 2 Failed: Single node ProductNode should remain unchanged"
    )

    # Test Case 3: Repeating ProductNode
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),)
    )
    assert result == expected, (
        "Test Case 3 Failed: Repeating nodes should be compressed"
    )

    # Test Case 4: Alternating ProductNode
    nodes = [
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
    ]

    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode(
        (
            RepeatNode(
                ProductNode(
                    (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(0)))
                ),
                CountValue(-3),
            ),
        )
    )
    assert result == expected, (
        "Test Case 4 Failed: Alternating pattern should be compressed with negative count"
    )

    # Test Case 5: Mixed ProductNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))]
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
        )
    )
    assert result == expected, (
        "Test Case 5 Failed: Mixed sequence compression incorrect"
    )

    # Test Case 6: SumNode Arithmetic Sequence
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]])
    )
    result = factorize_tuple(sum_node)
    expected_forward = SumNode(
        frozenset((RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(3)),))
    )
    expected_backward = SumNode(
        frozenset((RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(-3)),))
    )
    assert result in [expected_forward, expected_backward], (
        "Test Case 6 Failed: SumNode arithmetic sequence should be compressed"
    )

    # Test Case 7: SumNode Non-Sequence
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(i)) for i in [0, 5, 1]])
    )
    result = factorize_tuple(sum_node)
    assert result == sum_node, (
        "Test Case 7 Failed: Non-sequence SumNode should remain unchanged"
    )

    # Test Case 8: Non-Product/Sum Node
    node = PrimitiveNode(MoveValue(1))
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 8 Failed: Non-Product/Sum node should remain unchanged"
    )

    print("Test factorize_tuple - Passed")


def test_is_abstraction():
    """Tests the is_abstraction function for detecting VariableNodes in the tree."""
    # Test Case 1: Node is a VariableNode
    var_node = VariableNode(VariableValue(0))
    assert is_abstraction(var_node), (
        "Test Case 1 Failed: VariableNode itself should return True"
    )

    # Test Case 2: Node is a PrimitiveNode (no VariableNode)
    prim_node = PrimitiveNode(MoveValue(1))
    assert not is_abstraction(prim_node), (
        "Test Case 2 Failed: PrimitiveNode should return False"
    )

    # Test Case 3: ProductNode with one VariableNode child
    var_child = VariableNode(VariableValue(1))
    tuple_node = ProductNode((prim_node, var_child))
    assert is_abstraction(tuple_node), (
        "Test Case 3 Failed: ProductNode with VariableNode child should return True"
    )

    # Test Case 4: ProductNode with no VariableNodes
    tuple_no_var = ProductNode((prim_node, PrimitiveNode(MoveValue(2))))
    assert not is_abstraction(tuple_no_var), (
        "Test Case 4 Failed: ProductNode without VariableNodes should return False"
    )

    # Test Case 5: RepeatNode with VariableNode in subtree
    repeat_node = RepeatNode(var_child, CountValue(2))
    assert is_abstraction(repeat_node), (
        "Test Case 5 Failed: RepeatNode with VariableNode should return True"
    )

    # Test Case 6: SymbolNode with VariableNode as parameter
    symbol_with_var = SymbolNode(IndexValue(0), (var_child,))
    assert is_abstraction(symbol_with_var), (
        "Test Case 6 Failed: SymbolNode with VariableNode parameter should return True"
    )

    # Test Case 7: RootNode with VariableNode in program
    root_with_var = RootNode(
        var_child, CoordValue((0, 0)), PaletteValue(frozenset({1}))
    )
    assert is_abstraction(root_with_var), (
        "Test Case 7 Failed: RootNode with VariableNode in program should return True"
    )

    # Test Case 8: Node with only non-KNode BitLengthAware subvalues
    # PrimitiveNode has a Primitive subvalue (e.g., MoveValue), which is BitLengthAware but not a VariableNode
    assert not is_abstraction(prim_node), (
        "Test Case 8 Failed: Node with only non-KNode subvalues should return False"
    )

    print("Test is_abstraction - Passed")


def test_resolve_symbols():
    # Test Case 1: Test that a tree with no symbols remains unchanged.
    node = PrimitiveNode(MoveValue(1))
    symbols: tuple[KNode, ...] = ()
    result = resolve_symbols(node, symbols)
    assert result == node, "Tree with no symbols should remain unchanged"

    # Test Case 2: Test resolving a single SymbolNode without parameters.
    symbol_def = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    symbols = (symbol_def,)
    node = SymbolNode(IndexValue(0), ())
    result = resolve_symbols(node, symbols)
    assert result == symbol_def, (
        "SymbolNode should be replaced by its definition"
    )

    # Test Case 3: Test resolving a single SymbolNode with parameters.
    symbol_def = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(1)))
    )
    symbols = (symbol_def,)
    param = PrimitiveNode(MoveValue(4))
    node = SymbolNode(IndexValue(0), (param,))
    expected = ProductNode((param, PrimitiveNode(MoveValue(1))))
    result = resolve_symbols(node, symbols)
    assert result == expected, "Parameters should be substituted correctly"

    # Test Case 4: Test resolving a composite node containing a SymbolNode.
    symbol_def = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(1)))
    )
    symbols = (symbol_def,)
    param = PrimitiveNode(MoveValue(5))
    symbol_node = SymbolNode(IndexValue(0), (param,))
    composite = ProductNode((symbol_node, PrimitiveNode(MoveValue(6))))
    expected_inner = ProductNode((param, PrimitiveNode(MoveValue(1))))
    expected = ProductNode((expected_inner, PrimitiveNode(MoveValue(6))))
    result = resolve_symbols(composite, symbols)
    assert result == expected, (
        "Composite node should resolve its children correctly"
    )

    # Test Case 5: Test resolving nested symbols
    symbol0 = PrimitiveNode(MoveValue(7))
    symbol1 = ProductNode(
        (SymbolNode(IndexValue(0), ()), PrimitiveNode(MoveValue(8)))
    )
    symbols = (
        symbol0,
        symbol1,
    )
    node = SymbolNode(IndexValue(1), ())
    expected = ProductNode((symbol0, PrimitiveNode(MoveValue(8))))
    result = resolve_symbols(node, symbols)
    assert result == expected, "Nested symbols should be resolved recursively"

    # Test Case 6: Test resolving a symbol with parameters in a RepeatNode.
    symbol_def = RepeatNode(
        PrimitiveNode(MoveValue(2)), VariableNode(VariableValue(0))
    )
    symbols = (symbol_def,)
    param = CountValue(3)
    node = SymbolNode(IndexValue(0), (param,))
    expected = RepeatNode(PrimitiveNode(MoveValue(2)), param)
    result = resolve_symbols(node, symbols)
    assert result == expected, (
        "Parameters should substitute into RepeatNode's count"
    )

    # Test Case 7: Test that a SymbolNode with an invalid index remains unchanged.
    symbols = (PrimitiveNode(MoveValue(1)),)
    node = SymbolNode(IndexValue(1), ())
    result = resolve_symbols(node, symbols)
    assert result == node, (
        "SymbolNode with invalid index should remain unchanged"
    )

    # Test Case 8: Test resolving a tree with multiple symbols and parameters.
    symbol0 = PrimitiveNode(MoveValue(3))
    symbol1 = ProductNode(
        (SymbolNode(IndexValue(0), ()), VariableNode(VariableValue(0)))
    )
    symbols = (symbol0, symbol1)
    param = PrimitiveNode(MoveValue(4))
    node = SymbolNode(IndexValue(1), (param,))
    expected = ProductNode((symbol0, param))
    result = resolve_symbols(node, symbols)
    assert result == expected, (
        "Multiple symbols and parameters should be handled correctly"
    )
    print("Test resolve_symbols - Passed")


def test_find_symbol_candidates():
    """
    Tests the find_symbol_candidates function for identifying frequent subtrees
    across multiple Kolmogorov Trees, ensuring correct handling of frequency,
    bit-length savings, abstraction, edge cases, and integration.
    """

    # Test Case 1: Identical Repeating Pattern with Positive Bit Gain
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(2)))
    )  # bit_length = 15
    tree1 = RootNode(pattern, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    tree2 = RootNode(pattern, CoordValue((1, 1)), PaletteValue(frozenset({2})))
    tree3 = RootNode(pattern, CoordValue((2, 2)), PaletteValue(frozenset({3})))
    tree4 = RootNode(pattern, CoordValue((3, 3)), PaletteValue(frozenset({4})))
    trees = (tree1, tree2, tree3, tree4)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 1, (
        "Test Case 1 Failed: Should find exactly one candidate"
    )
    assert candidates[0] == RootNode(
        pattern, VariableNode(VariableValue(0)), VariableNode(VariableValue(1))
    ), "Test Case 1 Failed: Candidate should be the repeating pattern"
    print("Test Case 1: Basic Functionality - Passed")

    # Test Case 2: Frequency Threshold - Pattern Below Threshold
    unique_tree = RootNode(
        PrimitiveNode(MoveValue(3)),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    trees = (tree1, unique_tree)  # Pattern appears only once
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 0, (
        "Test Case 2 Failed: Should find no candidates below frequency threshold"
    )
    print("Test Case 2: Frequency Threshold - Passed")

    # Test Case 3: Bit-Length Savings - Exclude Non-Saving Patterns
    short_pattern = PrimitiveNode(MoveValue(2))  # Bit length too small to save
    tree5 = RootNode(
        short_pattern, CoordValue((0, 0)), PaletteValue(frozenset({1}))
    )
    tree6 = RootNode(
        short_pattern, CoordValue((1, 1)), PaletteValue(frozenset({2}))
    )
    trees = (tree5, tree6)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    # Short pattern (bit_length=6) won't save bits vs. SymbolNode (bit_length=10)
    assert len(candidates) == 0, (
        "Test Case 3 Failed: Should exclude patterns with no bit savings"
    )
    print("Test Case 3: Bit-Length Savings - Passed")

    # Test Case 4: Abstraction Handling - Abstracted Pattern
    tree7 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(1)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    tree8 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(3)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue((1, 1)),
        PaletteValue(frozenset({2})),
    )
    tree9 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(5)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue((2, 2)),
        PaletteValue(frozenset({3})),
    )
    trees = (tree7, tree8, tree9)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    expected_abs = ProductNode(
        (
            VariableNode(VariableValue(0)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
        )
    )
    assert any(candidate == expected_abs for candidate in candidates), (
        "Test Case 4 Failed: Should include abstracted pattern"
    )
    print("Test Case 4: Abstraction Handling - Passed")

    # Test Case 5a: Edge Case - Empty Input
    candidates = find_symbol_candidates((), min_occurrences=2, max_patterns=5)
    assert len(candidates) == 0, (
        "Test Case 5a Failed: Empty input should return empty list"
    )
    print("Test Case 5a: Edge Case (Empty Input) - Passed")

    # Test Case 5b: Edge Case - No Common Patterns
    tree10 = RootNode(
        PrimitiveNode(MoveValue(1)),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    tree11 = RootNode(
        PrimitiveNode(MoveValue(2)),
        CoordValue((1, 1)),
        PaletteValue(frozenset({2})),
    )
    trees = (tree10, tree11)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 0, (
        "Test Case 5b Failed: No common patterns should return empty list"
    )
    print("Test Case 5b: Edge Case (No Common Patterns) - Passed")

    # Test Case 6: Interaction with Other Functions - Symbolization and Resolution
    trees = (tree1, tree2, tree3, tree4)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=1
    )
    assert len(candidates) == 1, (
        "Test Case 6 Failed: Should find one candidate for symbolization"
    )
    symbol_index = IndexValue(0)
    symbolized_trees = tuple(
        abstract_node(symbol_index, candidates[0], tree) for tree in trees
    )
    symbols = (candidates[0],)
    resolved_trees = tuple(
        resolve_symbols(tree, symbols) for tree in symbolized_trees
    )
    assert resolved_trees == trees, (
        "Test Case 6 Failed: Resolved trees should match original trees"
    )
    print("Test Case 6: Interaction with Other Functions - Passed")

    print("All test_find_symbol_candidates tests - Passed")


def test_matching():
    # Test Case 1: No Variables
    node = PrimitiveNode(MoveValue(2))
    pattern = node
    bindings = matches(pattern, node)
    assert bindings == {}, (
        "Expected empty bindings for exact match without variables"
    )

    # Test Case 2: With Variables
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(3)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    expected_bindings = {0: PrimitiveNode(MoveValue(2))}
    assert bindings == expected_bindings, (
        f"Expected bindings {expected_bindings}, got {bindings}"
    )

    # Test Case 3: No match
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(4)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    assert bindings is None, "Expected no match, but got bindings"

    # Test Case 4: Abstraction and Resolution
    original_node = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(3)))
    )
    parameters = (PrimitiveNode(MoveValue(2)),)
    index = IndexValue(0)

    abstracted_node = abstract_node(index, pattern, original_node)
    assert isinstance(abstracted_node, SymbolNode), (
        "Expected SymbolNode after abstraction"
    )
    assert abstracted_node.index == index, "Index mismatch"
    assert abstracted_node.parameters == parameters, (
        f"Expected parameters {parameters}"
    )

    symbols = (pattern,)
    resolved_node = resolve_symbols(abstracted_node, symbols)
    assert resolved_node == original_node, (
        "Resolved node does not match original"
    )

    # Test Case 5: Nested Structures
    nested_node = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    pattern = ProductNode(
        (
            RepeatNode(VariableNode(VariableValue(0)), CountValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    parameters = (PrimitiveNode(MoveValue(2)),)
    index = IndexValue(0)

    abstracted_node = abstract_node(index, pattern, nested_node)
    assert isinstance(abstracted_node, SymbolNode), "Expected SymbolNode"
    assert abstracted_node.parameters == parameters, "Parameters mismatch"

    symbols = (pattern,)
    resolved_node = resolve_symbols(abstracted_node, symbols)
    assert resolved_node == nested_node, (
        "Resolved nested node does not match original"
    )

    # Tes Case 6: Multiple Variables
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(1)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    expected_bindings = {
        0: PrimitiveNode(MoveValue(2)),
        1: PrimitiveNode(MoveValue(3)),
    }
    assert bindings == expected_bindings, (
        f"Expected bindings {expected_bindings}, got {bindings}"
    )

    # Tes Case 7: Variable Reuse
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(0)))
    )
    subtree1 = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(2)))
    )
    bindings1 = matches(pattern, subtree1)
    assert bindings1 == {0: PrimitiveNode(MoveValue(2))}, (
        "Expected binding for identical elements"
    )

    subtree2 = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings2 = matches(pattern, subtree2)
    assert bindings2 is None, "Expected no match for different elements"

    # Test Case 8: Integration with generate_abstractions
    node = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    abstractions = generate_abstractions(node)

    for pattern, params in abstractions:
        index = IndexValue(0)
        abstracted_node = abstract_node(index, pattern, node)
        if abstracted_node != node:  # Abstraction occurred
            symbols = (pattern,)
            resolved_node = resolve_symbols(abstracted_node, symbols)
            assert resolved_node == node, (
                f"Resolved node does not match original for pattern {pattern}"
            )

    print("Test matching - Passed")


def test_factor_by_existing_symbols():
    """Tests the factor_by_existing_symbols function for replacing matching patterns with SymbolNodes."""
    # Test Case 1: Basic Pattern Replacement
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    tree = RootNode(pattern, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    symbols = (pattern,)
    result = factor_by_existing_symbols(tree, symbols)
    expected = RootNode(
        SymbolNode(IndexValue(0), (), pattern.bit_length()),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert result == expected, (
        "Test Case 1 Failed: Should replace pattern with SymbolNode"
    )
    print("Test Case 1: Basic Pattern Replacement - Passed")

    # Test Case 2: No Matching Pattern
    tree = RootNode(
        PrimitiveNode(MoveValue(1)),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    symbols = (pattern,)
    result = factor_by_existing_symbols(tree, symbols)
    assert result == tree, (
        "Test Case 2 Failed: Should return unchanged tree when no match"
    )
    print("Test Case 2: No Matching Pattern - Passed")

    # Test Case 3: Nested Pattern
    nested_tree = ProductNode((pattern, PrimitiveNode(MoveValue(4))))
    result = factor_by_existing_symbols(nested_tree, symbols)
    expected = ProductNode(
        (
            SymbolNode(IndexValue(0), (), pattern.bit_length()),
            PrimitiveNode(MoveValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 3 Failed: Should replace nested pattern"
    )
    print("Test Case 3: Nested Pattern - Passed")

    # Test Case 4: Multiple Matches
    multi_tree = ProductNode((pattern, pattern))
    result = factor_by_existing_symbols(multi_tree, symbols)
    symbol_node = SymbolNode(IndexValue(0), (), pattern.bit_length())
    expected = ProductNode((RepeatNode(symbol_node, CountValue(2)),))
    assert result == expected, (
        "Test Case 4 Failed: Should replace multiple occurrences"
    )
    print("Test Case 4: Multiple Matches - Passed")

    # Test Case 5: Empty Symbol Table
    result = factor_by_existing_symbols(tree, ())
    assert result == tree, (
        "Test Case 5 Failed: Should return unchanged tree with empty symbols"
    )
    print("Test Case 5: Empty Symbol Table - Passed")

    print("Test factor_by_existing_symbols - Passed")


def test_remap_symbol_indices():
    """Tests the remap_symbol_indices function for updating SymbolNode indices."""
    # Test Case 1: Single Symbol Remap
    tree = SymbolNode(IndexValue(0), ())
    mapping = [1]
    result = remap_symbol_indices(tree, mapping, 0)
    expected = SymbolNode(IndexValue(1), ())
    assert result == expected, "Test Case 1 Failed: Should remap index 0 to 1"
    print("Test Case 1: Single Symbol Remap - Passed")

    # Test Case 2: No Symbols
    tree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    mapping = [1, 2]
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 2 Failed: Should return unchanged tree with no symbols"
    )
    print("Test Case 2: No Symbols - Passed")

    # Test Case 3: Nested Symbols
    tree = ProductNode(
        (SymbolNode(IndexValue(0), ()), SymbolNode(IndexValue(1), ()))
    )
    mapping = [2, 0]
    result = remap_symbol_indices(tree, mapping, 0)
    expected = ProductNode(
        (SymbolNode(IndexValue(2), ()), SymbolNode(IndexValue(0), ()))
    )
    assert result == expected, (
        "Test Case 3 Failed: Should remap nested symbols correctly"
    )
    print("Test Case 3: Nested Symbols - Passed")

    # Test Case 4: Out of Bounds Index
    tree = SymbolNode(IndexValue(2), ())
    mapping = [0, 1]  # Mapping shorter than index
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 4 Failed: Should keep unchanged when index out of bounds"
    )
    print("Test Case 4: Out of Bounds Index - Passed")

    # Test Case 5: Empty Mapping
    tree = SymbolNode(IndexValue(0), ())
    mapping = []
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 5 Failed: Should keep unchanged with empty mapping"
    )
    print("Test Case 5: Empty Mapping - Passed")

    print("Test remap_symbol_indices - Passed")


def test_remap_sub_symbols():
    """Tests the remap_sub_symbols function for updating SymbolNodes within a symbol."""
    # Test Case 1: Simple Symbol with Sub-Symbol
    symbol = SymbolNode(IndexValue(0), (SymbolNode(IndexValue(1), ()),))
    mapping = [2, 0]
    original_table = (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(IndexValue(2), (SymbolNode(IndexValue(0), ()),))
    assert result == expected, (
        "Test Case 1 Failed: Should remap sub-symbol index"
    )
    print("Test Case 1: Simple Symbol with Sub-Symbol - Passed")

    # Test Case 2: No Sub-Symbols
    symbol = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(1)),))
    mapping = [1]
    original_table = (PrimitiveNode(MoveValue(2)),)
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(
        IndexValue(1), (PrimitiveNode(MoveValue(1)),), symbol.reference_length
    )
    assert result == expected, (
        "Test Case 2 Failed: Should remap outer symbol index"
    )
    print("Test Case 2: No Sub-Symbols - Passed")

    # Test Case 3: Multiple Sub-Symbols
    symbol = SymbolNode(
        IndexValue(0),
        (SymbolNode(IndexValue(0), ()), SymbolNode(IndexValue(1), ())),
    )
    mapping = [1, 0]
    original_table = (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(
        IndexValue(1),
        (SymbolNode(IndexValue(1), ()), SymbolNode(IndexValue(0), ())),
    )
    assert result == expected, (
        "Test Case 3 Failed: Should remap multiple sub-symbols"
    )
    print("Test Case 3: Multiple Sub-Symbols - Passed")

    # Test Case 4: Empty Mapping
    symbol = SymbolNode(IndexValue(0), (SymbolNode(IndexValue(0), ()),))
    mapping = []
    original_table = (PrimitiveNode(MoveValue(1)),)
    result = remap_sub_symbols(symbol, mapping, original_table)
    assert result == symbol, (
        "Test Case 4 Failed: Should return unchanged with empty mapping"
    )
    print("Test Case 4: Empty Mapping - Passed")

    print("Test remap_sub_symbols - Passed")


def test_merge_symbol_tables():
    """Tests the merge_symbol_tables function for combining symbol tables."""
    # Test Case 1: Merging Identical Tables
    symbol = ProductNode(
        (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    )
    table1 = (symbol,)
    table2 = (symbol,)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == (symbol,), "Test Case 1 Failed: Unified table incorrect"
    assert mappings == [[0], [0]], (
        "Test Case 1 Failed: Mappings should point to same index"
    )
    print("Test Case 1: Merging Identical Tables - Passed")

    # Test Case 2: Merging Distinct Tables
    symbol1 = PrimitiveNode(MoveValue(1))
    symbol2 = PrimitiveNode(MoveValue(2))
    table1 = (symbol1,)
    table2 = (symbol2,)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == (symbol1, symbol2), (
        "Test Case 2 Failed: Unified table should contain both symbols"
    )
    assert mappings == [[0], [1]], "Test Case 2 Failed: Mappings incorrect"
    print("Test Case 2: Merging Distinct Tables - Passed")

    # Test Case 3: Merging with Nested Symbols
    nested_symbol = RepeatNode(SymbolNode(IndexValue(0), ()), CountValue(3))
    symbol_def = PrimitiveNode(MoveValue(1))
    table1 = (symbol_def, nested_symbol)
    table2 = (symbol_def,)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == table1, "Test Case 3 Failed: Unified table incorrect"
    assert mappings == [[0, 1], [0]], "Test Case 3 Failed: Mappings incorrect"
    print("Test Case 3: Merging with Nested Symbols - Passed")

    # Test Case 4: Merging with Nested Symbols and Variables
    resolved = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(4)), PrimitiveNode(MoveValue(5)))),
        CountValue(3),
    )
    symbol1 = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(4)), PrimitiveNode(MoveValue(5)))),
        VariableNode(VariableValue(0)),
    )
    symbol2 = SymbolNode(IndexValue(0), (CountValue(3),))
    table1 = (resolved,)
    table2 = (symbol1, symbol2)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == table2, "Test Case 3 Failed: Unified table incorrect"
    assert mappings == [[1], [0, 1]], "Test Case 3 Failed: Mappings incorrect"
    print("Test Case 3: Merging with Nested Symbols - Passed")

    # Test Case 4: Empty Tables
    unified, mappings = merge_symbol_tables([(), ()])
    assert unified == (), (
        "Test Case 4 Failed: Should return empty unified table"
    )
    assert mappings == [[], []], (
        "Test Case 4 Failed: Should return empty mappings"
    )
    print("Test Case 4: Empty Tables - Passed")

    print("Test merge_symbol_tables - Passed")


def test_symbolize_together():
    """Tests the symbolize_together function for unifying and re-symbolizing trees."""
    # Test Case 1: Identical Trees with Empty Symbol Tables
    pattern = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    tree1 = RootNode(pattern, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    tree2 = RootNode(pattern, CoordValue((1, 1)), PaletteValue(frozenset({2})))
    trees = (tree1, tree2)
    symbol_tables = [(), ()]
    final_trees, final_symbols = symbolize_together(trees, symbol_tables)
    expected_symbol = RootNode(
        node=ProductNode(
            children=(
                PrimitiveNode(value=MoveValue(value=2)),
                PrimitiveNode(value=MoveValue(value=3)),
                PrimitiveNode(value=MoveValue(value=4)),
                PrimitiveNode(value=MoveValue(value=4)),
            )
        ),
        position=VariableNode(index=VariableValue(value=0)),
        colors=VariableNode(index=VariableValue(value=1)),
    )
    expected_trees = (
        SymbolNode(
            index=IndexValue(value=0),
            parameters=(
                CoordValue(value=(0, 0)),
                PaletteValue(value=frozenset({1})),
            ),
            reference_length=expected_symbol.bit_length(),
        ),
        SymbolNode(
            index=IndexValue(value=0),
            parameters=(
                CoordValue(value=(1, 1)),
                PaletteValue(value=frozenset({2})),
            ),
            reference_length=expected_symbol.bit_length(),
        ),
    )
    expected_symbols = (expected_symbol,)
    assert final_trees == expected_trees, "Test Case 1 Failed: Trees incorrect"
    assert final_symbols == expected_symbols, (
        "Test Case 1 Failed: Symbols incorrect"
    )
    print("Test Case 1: Identical Trees with Empty Symbol Tables - Passed")

    # Test Case 2: Trees with Pre-existing Symbols
    symbol_def = PrimitiveNode(MoveValue(1))
    tree1 = SymbolNode(IndexValue(0), ())
    tree2 = SymbolNode(IndexValue(0), ())
    symbol_tables = [(symbol_def,), (symbol_def,)]
    final_trees, final_symbols = symbolize_together(
        (tree1, tree2), symbol_tables
    )
    assert final_trees == (tree1, tree2), (
        "Test Case 2 Failed: Trees should remain unchanged"
    )
    assert final_symbols == (symbol_def,), (
        "Test Case 2 Failed: Symbols should be unified"
    )
    print("Test Case 2: Trees with Pre-existing Symbols - Passed")

    # Test Case 3: Empty Input
    final_trees, final_symbols = symbolize_together((), [])
    assert final_trees == (), (
        "Test Case 3 Failed: Empty trees should return empty"
    )
    assert final_symbols == (), (
        "Test Case 3 Failed: Empty symbols should return empty"
    )
    print("Test Case 3: Empty Input - Passed")

    # Test Case 4: Integration Test with Resolution
    pattern = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    tree1 = RootNode(pattern, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    tree2 = RootNode(pattern, CoordValue((1, 1)), PaletteValue(frozenset({2})))
    trees = (tree1, tree2)
    symbol_tables = [(), ()]  # Empty symbol tables to start fresh
    final_trees, final_symbols = symbolize_together(trees, symbol_tables)
    resolved_trees = tuple(
        resolve_symbols(tree, final_symbols) for tree in final_trees
    )
    expected_resolved = (
        tree1,
        tree2,
    )  # Expect the original trees after resolution
    assert resolved_trees == expected_resolved, (
        "Test Case 4 Failed: Resolved trees should match originals"
    )
    print("Test Case 4: Integration Test with Resolution - Passed")

    print("Test symbolize_together - Passed")


def run_tests():
    """Runs simple tests to verify KolmogorovTree functionality."""
    # Test 1: Basic node creation and bit length
    move_right = PrimitiveNode(MoveValue(2))
    assert move_right.bit_length() == 6, (
        "PrimitiveNode bit length should be 6 (3 + 3)"
    )
    product = ProductNode((move_right, move_right))
    assert product.bit_length() == 17, (
        "ProductNode bit length should be 17 (3 + 6 + 6 + 2)"
    )
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode(frozenset((move_right, move_down)))
    assert sum_node.bit_length() == 17, (
        "SumNode bit length should be 17 (3 + 2 + 6 + 6)"
    )
    repeat = RepeatNode(move_right, CountValue(3))
    assert repeat.bit_length() == 14, (
        "RepeatNode bit length should be 14 (3 + 6 + 5)"
    )
    symbol = SymbolNode(IndexValue(0), ())
    assert symbol.bit_length() == 10, (
        "SymbolNode bit length should be 10 (3 + 7)"
    )
    root = RootNode(product, CoordValue((0, 0)), PaletteValue(frozenset({1})))
    assert root.bit_length() == 34, (
        "RootNode bit length should be 34 (3 + 17 + 10 + 4)"
    )
    print("Test 1: Basic node creation and bit length - Passed")

    # Test 2: String representations
    assert str(move_right) == "2", "PrimitiveNode str should be '2'"
    assert str(product) == "22", "ProductNode str should be '22'"
    assert str(sum_node) == "[2|3]", "SumNode str should be '[2|3]'"
    assert str(repeat) == "(2)*{3}", "RepeatNode str should be '(2)*{3}'"
    assert str(symbol) == "s_0", "SymbolNode str should be 's_0'"
    assert str(root) == "Root(22, (0, 0), {1})", (
        "RootNode str should be 'Root(22, (0, 0), {1})'"
    )
    print("Test 2: String representations - Passed")

    # Test 3: Operator overloads
    assert (move_right | move_down) == SumNode(
        frozenset((move_right, move_down))
    ), "Operator | failed"
    assert (move_right & move_down) == ProductNode((move_right, move_down)), (
        "Operator & failed"
    )
    assert (move_right + move_down) == ProductNode((move_right, move_down)), (
        "Operator + failed"
    )
    assert (move_right * 3) == RepeatNode(move_right, CountValue(3)), (
        "Operator * failed"
    )
    assert ((move_right * 2) * 3) == RepeatNode(move_right, CountValue(6)), (
        "Nested * failed"
    )
    print("Test 3: Operator overloads - Passed")

    # Test 4: Symbol resolution
    symbol_def = ProductNode((move_right, move_down, move_right))
    symbols: tuple[KNode[MoveValue], ...] = (symbol_def,)
    symbol_node = SymbolNode(IndexValue(0), ())
    root_with_symbol = RootNode(
        symbol_node,
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    resolved = resolve_symbols(root_with_symbol, symbols)
    expected_resolved = RootNode(
        ProductNode((move_right, move_down, move_right)),
        CoordValue((0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

    test_encode_run_length()
    test_construct_product_node()
    test_shift()
    test_get_iterator()

    test_find_repeating_pattern()
    test_factorize_tuple()
    test_is_abstraction()
    test_resolve_symbols()
    test_find_symbol_candidates()
    test_matching()
    test_factor_by_existing_symbols()
    test_remap_symbol_indices()
    test_remap_sub_symbols()
    test_merge_symbol_tables()
    test_symbolize_together()


if __name__ == "__main__":
    run_tests()
    print("All tests passed successfully!")
