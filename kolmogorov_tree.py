"""
Kolmogorov Tree: An abstract-syntax tree structure forming to represent non-deterministic bit-length-aware programs.
TL;DR: An bitlengthaware AST for bitlengthaware rosetrees

This module implements a tree structure for representing non-deterministic programs,
for example to describe shapes using a non-deterministic program representing 2D grid movements (see syntax_tree.py). The representation
aims to approximate Kolmogorov complexity through minimum description length.

The tree structure consists of:
- ProductNodes for deterministic sequences
- SumNodes for non-deterministic branching
- RepeatNodes for repetition extraction for ProductNodes, an artifact of DFS
- NestedNodes is like RepeatNodes but for SumNodes, an artificat of BFS
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
import itertools

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Set
from dataclasses import dataclass, field
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

from localtypes import (
    BitLengthAware,
    Color,
    Coord,
    Primitive,
    ensure_all_instances,
)
from utils.tree_functionals import (
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
    NODE_TYPE = 4  # 4 bits for up to 16 node types
    INDEX = 7  # 7 bits for symbol indices (up to 128 symbols)
    VAR = 1  # 2 bits for variable indices (up to 2 variables per symbol)
    NONE = 0


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


@dataclass(frozen=True)
class NoneValue(Primitive):
    """Represents a None."""

    value: None = None

    def bit_length(self) -> int:
        return BitLength.NONE


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

    def __str__(self) -> str:
        return str(tuple(self.value))


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
        #
        children_str = [str(child) for child in self.children]  # Compute once
        sorted_str = sorted(children_str)  # Sort precomputed strings
        return "[" + "|".join(sorted_str) + "]"  # Use precomputed strings


@dataclass(frozen=True)
class RepeatNode(KNode[T]):
    """Represents repetition of a node a specified number of times."""

    node: KNode[T]
    count: CountValue | VariableNode  # Count can be fixed or parameterized

    def bit_length(self) -> int:
        count_len = self.count.bit_length()
        return super().bit_length() + self.node.bit_length() + count_len

    def __str__(self) -> str:
        return f"({str(self.node)})*{{{self.count}}}"


# TO-DO for Nested node:
# [ ] Extraction
# [ ] Decoding
# [ ] Symbolization
@dataclass(frozen=True)
class NestedNode(KNode[T]):
    """
    Represents A finite equivalent of the fixed point combinator.
    The catch is it can only acts on possibly recursive properties.
    It acts like a more powerful version of the SymbolNode for abstracted patterns.
    It looks like RepeatNode a lot, especially because its kind of the same for breadth-first traversal.
    It can be seen as a RepeatNode for Symbols

    Y_0(i, node) = node
    Y_c(i, node) ~ "s_i((Y_c-1(i, node),))"
    """

    index: IndexValue
    node: KNode[T]
    count: CountValue | VariableNode

    def bit_length(self) -> int:
        index_len = self.index.bit_length()
        terminal_node = self.node.bit_length()
        count_len = self.count.bit_length()
        return super().bit_length() + index_len + terminal_node + count_len

    def __str__(self) -> str:
        return f"Y_{{{self.count}}}({self.index}, {self.node})"


@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """Represents an abstraction or reusable pattern."""

    index: IndexValue  # Index in the symbol table
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_len

    def __str__(self) -> str:
        if self.parameters:
            return (
                f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
            )
        return f"s_{self.index} "


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
class RootNode(KNode[T]):
    """Root node: it wraps the program's node with its starting context."""

    node: KNode[T] | NoneValue
    position: CoordValue | VariableNode  # Starting position
    colors: PaletteValue | VariableNode  # Colors used in the shape

    def bit_length(self) -> int:
        pos_len = self.position.bit_length()
        colors_len = self.colors.bit_length()
        return (
            super().bit_length() + self.node.bit_length() + pos_len + colors_len
        )

    def __str__(self) -> str:
        position = self.position
        pos_str = str(position)
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


def extract_template(knode: KNode[T]) -> list[tuple[KNode[T], Parameters]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions: list[tuple[KNode[T], Parameters]] = []

    # If the node is already a lambda-abstraction
    # It should not be abstracted further
    if is_abstraction(knode):
        return abstractions

    match knode:
        case ProductNode(children) if len(children) > 2:
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
            abstractions.append((ProductNode(nodes1), (max_children[0],)))

            # If there are at least two distinct children and length > 2
            if len(max_children) > 1 and len(children) > 2:
                # Abstract the second most frequent/largest child
                nodes2 = tuple(
                    VariableNode(VariableValue(0))
                    if c == max_children[1]
                    else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes2), (max_children[0],)))

                # Then absttract the top two
                nodes3 = tuple(
                    VariableNode(VariableValue(max_children.index(c)))
                    if c in max_children
                    else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes3), tuple(max_children)))
        case SumNode(children) if len(children) > 2:
            # Abstract up to two distinct elements for now
            child_counter = Counter(children)  # Count occurrences of each child
            child_set = list(child_counter.keys())
            max_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]

            # Abstract the most frequent/largest child
            nodes1 = frozenset(
                VariableNode(VariableValue(0)) if c == max_children[0] else c
                for c in children
            )
            abstractions.append((SumNode(nodes1), (max_children[0],)))

            # If there are at least two distinct children and length > 2
            if len(max_children) > 1 and len(children) > 2:
                # Abstract the second most frequent/largest child
                nodes2 = frozenset(
                    VariableNode(VariableValue(0))
                    if c == max_children[1]
                    else c
                    for c in children
                )
                abstractions.append((SumNode(nodes2), (max_children[0],)))

                # Then absttract the top two
                nodes3 = frozenset(
                    VariableNode(VariableValue(max_children.index(c)))
                    if c in max_children
                    else c
                    for c in children
                )
                abstractions.append((SumNode(nodes3), tuple(max_children)))
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
        case NestedNode(index, node, count):
            abstractions.extend(
                [
                    (
                        NestedNode(
                            index, VariableNode(VariableValue(0)), count
                        ),
                        (node,),
                    ),
                    (
                        NestedNode(
                            index,
                            node,
                            VariableNode(VariableValue(0)),
                        ),
                        (count,),
                    ),
                    (
                        NestedNode(
                            index,
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(1)),
                        ),
                        (
                            node,
                            count,
                        ),
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


def extract_nested_sum_template(
    snode: SumNode[T],
) -> tuple[SumNode[T], SumNode[T]] | None:
    """
    Like generate abstractions but for structures like [0[]|4]

    Returns
        tuple[SumNode[T], SumNode[T]]: (template, parameter)
    """
    max_product = max(
        [node for node in snode.children if isinstance(node, ProductNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_product:
        return None

    max_sum = max(
        [node for node in max_product.children if isinstance(node, SumNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_sum:
        return None

    new_product = ProductNode(
        tuple(
            node if node != max_sum else VariableNode(VariableValue(0))
            for node in max_product.children
        )
    )

    new_sum = SumNode(
        frozenset(
            {
                node if node != max_product else new_product
                for node in snode.children
            }
        )
    )

    return new_sum, max_sum


def extract_nested_product_template(
    pnode: ProductNode[T],
) -> tuple[ProductNode[T], ProductNode[T]] | None:
    """
    Like generate abstractions but for structures like [0[]|4]

    Returns
        tuple[SumNode[T], SumNode[T]]: (template, parameter)
    """
    max_sum = max(
        [node for node in pnode.children if isinstance(node, SumNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_sum:
        return None

    max_product = max(
        [node for node in max_sum.children if isinstance(node, ProductNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_product:
        return None

    new_sum = SumNode(
        frozenset(
            {
                node if node != max_product else VariableNode(VariableValue(0))
                for node in max_sum.children
            }
        )
    )

    new_product = ProductNode(
        tuple(node if node != max_sum else new_sum for node in pnode.children)
    )

    return new_product, max_product


def detect_recursive_collection(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[KNode[T], KNode[T], int] | None:
    """
    Detects a recursive pattern in a SumNode, returning the common template, terminal node, and recursion count.

    Args:
        snode: The SumNode to analyze.

    Returns:
        tuple[KNode[T], KNode[T], int]: (template, terminal_node, count) if a recursive pattern is found, else None.
    """
    count = 0
    current = knode
    common_template = None

    while True:
        assert isinstance(current, type(knode))
        extraction = (
            extract_nested_sum_template(current)
            if isinstance(current, SumNode)
            else extract_nested_product_template(current)
        )

        if extraction is None:
            break
        template, parameter = extraction

        if count == 0:
            common_template = (
                template  # Set the template on the first iteration
            )
        elif template != common_template:
            break  # Stop if the template changes

        if not isinstance(parameter, type(knode)):
            break  # Stop if the parameter isn’t a SumNode

        count += 1
        current = parameter

    if count >= 1 and common_template is not None:
        return common_template, current, count
    return None


def nested_collection_to_nested_node(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[NestedNode[T], KNode[T]] | None:
    """
    Transforms a CollectionNode with a recursive pattern into a NestedNode, returning the node and its template.

    Args:
        snode: The SumNode to transform.

    Returns:
        tuple[NestedNode[T], KNode[T]]: (nested_node, template) if a recursive pattern is found, else None.
    """
    recursive_info = detect_recursive_collection(knode)
    if recursive_info is None:
        return None

    template, terminal_node, count = recursive_info
    # Use a placeholder index; the symbol table must be updated externally
    nested_node = NestedNode(
        index=IndexValue(0), node=terminal_node, count=CountValue(count)
    )
    return nested_node, template


def get_iterator_old(knodes: Collection[KNode[T]]) -> frozenset[KNode[T]]:
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
    ) or all(isinstance(n, RepeatNode) for n in nodes):
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
                    if isinstance(encoded, ProductNode):
                        simplified[i:j] = list(
                            encoded.children
                        )  # Insert the children directly                    i += 1
                        i += len(
                            encoded.children
                        )  # Move index past inserted elements
                    else:
                        simplified[i:j] = [encoded]
                        i += 1
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


def iterable_to_product(iterable: Iterable[KNode[T]]) -> KNode[T]:
    nodes: list[KNode[T]] = list(iterable)
    if not nodes:
        raise ValueError("Empty iterable. Can't produce a product out of it")
    elif len(nodes) == 1:
        return nodes[0]
    else:
        nnodes = construct_product_node(nodes)
        return flatten_product(nnodes)


def iterable_to_sum(
    iterable: Iterable[KNode[T]], compress_iterator=False
) -> KNode[T] | None:
    nodes: frozenset[KNode[T]] = frozenset(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return next(iter(nodes))
    elif compress_iterator:
        return SumNode(get_iterator(nodes))
    else:
        return SumNode(nodes)


def reconstruct_knode(
    knode: KNode[T], new_children: Sequence[KNode[T]], factorize: bool = False
) -> KNode[T]:
    """Reconstructs a KNode with its original data and new children."""
    match knode:
        case SumNode(_):
            new_node = SumNode(frozenset(new_children))
            if factorize:
                new_node = factorize_tuple(new_node)
            return new_node
        case ProductNode(_):
            new_node = ProductNode(tuple(new_children))
            if factorize:
                new_node = factorize_tuple(new_node)
            return new_node
        case RepeatNode(_, count):
            return RepeatNode(new_children[0], count)
        case NestedNode(index, _, count):
            return NestedNode(index, new_children[0], count)
        case RootNode(_, position, colors):
            if not new_children:
                return RootNode(NoneValue(), position, colors)
            return RootNode(new_children[0], position, colors)
        case SymbolNode(index, parameters):
            new_parameters = list(parameters)
            for i, p in enumerate(new_children):
                new_parameters[i] = p
            return SymbolNode(index, tuple(new_parameters))
        case _:
            return knode  # Leaf nodes remain unchanged


# Traversal


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


def children(knode: KNode) -> Iterator[KNode]:
    """Unified API to access children of standard KNodes nodes"""
    subvalues = get_subvalues(knode)
    return (sv for sv in subvalues if isinstance(sv, KNode))


# Traversal functions


# bitlength aware
# def breadth_first_preorder_bitlengthaware(
#     root: BitLengthAware,
# ) -> Iterator[BitLengthAware]:
#     return breadth_first_preorder(get_subvalues, root)


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
        case NestedNode(index, node, count):
            return NestedNode(index, reverse_node(node), count)
        case SymbolNode(index, parameters):
            reversed_params = tuple(
                reverse_node(p) if isinstance(p, KNode) else p
                for p in reversed(parameters)
            )
            return SymbolNode(index, reversed_params)
        case RootNode(node, position, colors):
            nnode = (
                NoneValue()
                if isinstance(node, NoneValue)
                else reverse_node(node)
            )
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


def encode_run_length(
    primitives: Iterable[PrimitiveNode[T]],
) -> ProductNode[T] | RepeatNode[T] | PrimitiveNode[T]:
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
    if len(sequence) == 1:
        assert isinstance(sequence[0], RepeatNode | PrimitiveNode)
        return sequence[0]

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
    subvalues = depth_first_preorder_bitlengthaware(node)
    variable_numbers = [
        value.value for value in subvalues if isinstance(value, VariableValue)
    ]
    if variable_numbers:
        return max(variable_numbers) + 1
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
            iterable_to_product(pattern) if len(pattern) > 1 else pattern[0]
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
                        iterable_to_product(segment)
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

    # Preserves iterators
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
        first = next(iter(node.children))
        if (
            isinstance(first, RepeatNode)
            and isinstance(first.count, CountValue)
            and first.count == 5
        ):
            print("TEST")
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
    return cast(KNode, iterable_to_product(simplified))


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


# Decompression
def expand_repeats(
    node: KNode[T],
) -> KNode[T]:
    """Used to uncompress knodes"""

    def expand_repeat_f(knode: KNode) -> KNode:
        """Expand repeats"""
        if isinstance(knode, RepeatNode):
            if isinstance(knode.count, VariableNode):
                raise TypeError("Trying to uncompress a variable repeat")
            return cast(KNode, ProductNode((knode.node,) * knode.count.value))
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


def unify_sum_children(
    p_idx: int, # Index of the current pattern child being matched
    pattern_list: list[KNode[T]], # Sorted list of pattern children
    subtree_list: list[KNode[T]], # Sorted list of subtree children
    subtree_used: list[bool],     # Tracks which subtree children are already matched
    bindings: Bindings
) -> Bindings | None:
    """
    Helper function using backtracking on sorted lists to find a deterministic
    unification mapping between SumNode children.

    Args:
        p_idx: Current index in pattern_list being processed.
        pattern_list: Sorted list of pattern children.
        subtree_list: Sorted list of subtree children.
        subtree_used: Boolean mask indicating used subtree children.
        bindings: Current accumulated bindings for this search path.

    Returns:
        Updated bindings if a valid matching is found, otherwise None.
    """
    # Base case: All pattern children have been successfully matched.
    if p_idx == len(pattern_list):
        return bindings

    p_child = pattern_list[p_idx]

    # Iterate through potential matches in the sorted subtree list
    for s_idx in range(len(subtree_list)):
        # Check if this subtree child is already used in the current mapping
        if not subtree_used[s_idx]:
            s_child = subtree_list[s_idx]

            # Create a copy of bindings to explore this specific pair's unification
            current_bindings = bindings.copy()

            # Attempt to unify the chosen pair (p_child, s_child)
            if unify(p_child, s_child, current_bindings):
                # If the pair unifies successfully:
                # 1. Mark the subtree child as used for this path
                subtree_used[s_idx] = True

                # 2. Recursively try to match the *next* pattern child (p_idx + 1)
                result_bindings = unify_sum_children(
                    p_idx + 1,
                    pattern_list,
                    subtree_list,
                    subtree_used,
                    current_bindings  # Pass the updated bindings
                )

                # 3. Check if the recursive call was successful
                if result_bindings is not None:
                    # Success! A complete valid matching was found. Return it.
                    return result_bindings

                # 4. Backtrack: If the recursive call failed, unmark the
                #    subtree child so it can be tried with other pattern children
                #    in alternative branches of the search.
                subtree_used[s_idx] = False
                # Continue the loop to try matching p_child with the next unused s_child


    # If the loop completes without returning, it means p_child could not be
    # successfully matched with any *available* s_child in a way that allowed the
    # rest of the pattern children to be matched. Backtrack by returning None.
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
            assert isinstance(subtree, Primitive)
            return value == subtree.value
            # Inside the unify function...
        case SumNode(children):
            assert isinstance(subtree, SumNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            if not children:
                return True

            # 1. Create sorted lists based on a canonical key
            # Sort using the  string representation
            sorted_pattern_children = sorted(children, key=str)
            sorted_subtree_children = sorted(s_children, key=str)
            # 2. Initialize the tracking list for used subtree children
            subtree_used = [False] * len(sorted_subtree_children)

            # 3. Call the deterministic backtracking helper
            initial_bindings = bindings.copy()
            successful_bindings = unify_sum_children(
                0, # Start matching the first pattern child (index 0)
                sorted_pattern_children,
                sorted_subtree_children,
                subtree_used,
                initial_bindings
            )

            if successful_bindings is not None:
                # Update original bindings *in place*
                bindings.clear()
                bindings.update(successful_bindings)
                return True
            else:
                return False

        case ProductNode(children):
            assert isinstance(subtree, ProductNode)
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
        case NestedNode(index, node, count):
            subtree = cast(NestedNode, subtree)
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
        return SymbolNode(index, ())

    bindings = matches(pattern, node)

    if bindings is not None:
        params = tuple(
            binding
            for j in range(max(bindings.keys(), default=-1) + 1)
            if (binding := bindings.get(j, None)) is not None
        )
        return SymbolNode(index, params)

    return node


def node_to_symbolized_node(
    index: IndexValue, pattern: KNode[T], knode: KNode[T]
) -> KNode[T]:
    return postmap(knode, lambda node: abstract_node(index, pattern, node))


def substitute_variables_deprecated(
    abstraction: KNode[T], params: Parameters
) -> KNode[T]:
    """
    Helper function to substitute variables, including those that are not traversed
    during a simple node traversal
    """
    print(f"abstraction: {abstraction}, params: {params}")
    match abstraction:
        case VariableNode(index) if index.value < len(params):
            node = params[index.value]
            if not isinstance(node, KNode):
                raise TypeError(
                    f"Trying to substitute a non-node parameter to a variable encountered during node traversal: {[node]}"
                )
            return node
        case RepeatNode(node, VariableNode(index)) if index.value < len(params):
            count = params[index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return RepeatNode(node, count)
        case NestedNode(index, node, count) if isinstance(
            count, VariableNode
        ) and count.index.value < len(params):
            count = params[count.index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return NestedNode(index, node, count)
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


def variable_to_param(node: VariableNode, params: Parameters) -> BitLengthAware:
    if node.index.value < len(params):
        value = params[node.index.value]
        return value
    return node
    # raise ValueError(f"Unknown variable {node.index.value} for {params}")


def substitute_variables(knode: KNode[T], params: Parameters) -> KNode[T]:
    # Avoid infinite recursion on VariableNodes by treating them immediately
    # And returning directly
    if isinstance(knode, VariableNode):
        node = variable_to_param(knode, params)
        if not isinstance(node, KNode):
            raise TypeError(f"{node} is not a KNode")
        return node

    # Iterating over fields
    nfields = {}
    for field_name, value in vars(knode).items():
        match value:
            # Replace directly variables to avoid hitting the TypeError
            case VariableNode():
                nvalue = variable_to_param(value, params)
            case frozenset():
                nvalue = frozenset(
                    {substitute_variables(node, params) for node in value}
                )
            case tuple():
                nvalue = tuple(
                    substitute_variables(node, params) for node in value
                )
            case KNode():
                nvalue = substitute_variables(value, params)
            case _:
                nvalue = value

        nfields[field_name] = nvalue

    return type(knode)(**nfields)

@functools.cache
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
    # return postmap(abstraction, lambda node: substitute_variables(node, params))
    return substitute_variables(abstraction, params)


@functools.cache
def resolve_symbols(knode: KNode[T], symbols: Sequence[KNode[T]]) -> KNode[T]:
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

def create_variable_node(i: int) -> VariableNode:
    """Creates a VariableNode(VariableValue)."""
    # Assuming VariableValue is defined similarly to MoveValue
    return VariableNode(VariableValue(i))

def cv(r: int, c: int) -> CoordValue:
     """ Creates a CoordValue. """
     return CoordValue(Coord(c, r)) # Assuming Coord(col, row)

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


def extract_nested_patterns(
    symbol_table: list[KNode[T]],
    tree: KNode[T],
) -> KNode[T]:
    """
    Traverses a Kolmogorov Tree to identify and replace nested patterns with NestedNodes,
    extracting their templates into a symbol table.

    Args:
        tree: The input Kolmogorov Tree (KNode[T]).

    Returns:
        KNode[T]: The transformed tree with NestedNodes replacing nested patterns.

    Edge-Effects:
        Fill the given symbol table with templates extracted from the nested patterns.
    """

    def mapping_function(node: KNode[T]) -> KNode[T]:
        """
        Mapping function for postmap to detect nested patterns and replace them with NestedNodes.

        Args:
            node: The current node being processed.

        Returns:
            KNode[T]: The transformed node (either a NestedNode or the original node).
        """
        # Only process SumNode or ProductNode for nested patterns
        if isinstance(node, (SumNode, ProductNode)):
            result = nested_collection_to_nested_node(node)
            if result is not None:
                nested_node, template = result
                # Greedy test, it might be better because the template might be shared with others
                if (
                    nested_node.bit_length() + template.bit_length()
                    <= node.bit_length()
                ):
                    # Add the template to the symbol table and get its index
                    if template in symbol_table:
                        index = symbol_table.index(template)
                    else:
                        index = len(symbol_table)
                        symbol_table.append(template)
                    # Create a new NestedNode with the correct index
                    # Since NestedNode is frozen, instantiate a new one
                    new_nested_node = NestedNode(
                        IndexValue(index), nested_node.node, nested_node.count
                    )
                    return new_nested_node
        # Return the node unchanged if no pattern is detected
        return node

    # Apply postmap to transform the tree
    new_tree = premap(tree, mapping_function)
    return new_tree


def expand_nested_node(
    nested_node: NestedNode[T], symbol_table: Sequence[KNode[T]]
) -> KNode[T]:
    """Expands a NestedNode using its template from the symbol table."""
    template = symbol_table[nested_node.index.value]
    current = nested_node.node

    if not isinstance(nested_node.count, CountValue):
        raise ValueError(
            f"Trying to expand a nested node with a variable count: {nested_node}"
        )

    for _ in range(nested_node.count.value):
        current = postmap(
            template, lambda node: substitute_variables(node, (current,))
        )
    return current


def expand_all_nested_nodes(knode: KNode[T], symbol_table: Sequence[KNode[T]]):
    """
    Expand all the inner NestedNodes of a given node.

    Args:
        knode: KNode possibly containing nested nodes
        symbol_table: Symbol Table containing all the template referenced by NestedNode. It will fail otherwise.
    Returns:
        KNode[T]: node free of NestedNodes
    """

    def expand_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, NestedNode):
            return expand_nested_node(node, symbol_table)
        return node

    return postmap(knode, expand_f)


def find_symbol_candidates(
    trees: Sequence[KNode[T]],
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
        list
    )  # Maps pattern to set of matched subtrees

    for subtree in all_subtrees:
        pattern_counter[subtree] += 1
        pattern_matches[subtree].append(subtree)
        for abs_pattern, params in extract_template(subtree):
            pattern_counter[abs_pattern] += 1
            pattern_matches[abs_pattern].append(subtree)

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
                for _, ps in extract_template(s)
                if pat == _
                for p in ps
            )
            / count
        )
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        # return (count - 1) * (avg_len - symb_len) - pat.bit_length()
        return current_len - (count * symb_len + pat.bit_length())

    """
    # TO-DO: Currently repeats of simple moves by the same count of a move are not symbolized,
    # because the SymbolNode is heavier
    # so a lot of repeat by the same count won't be symbolize
    """
    common_patterns_test = [
        pat for pat in common_patterns if bit_gain(pat) > 1000
    ]
    for pat in common_patterns_test:
        print(f"\n\nPattern: {pat}")
        print(f"Bit gain: {bit_gain(pat)}")
        print(f"Count: {pattern_counter[pat]}")
        print(
            f"Current len: {sum(s.bit_length() for s in pattern_matches[pat])}"
        )
        print(
            f"Param len: {
                (
                    sum(
                        p.bit_length()
                        for s in pattern_matches[pat]
                        for _, ps in extract_template(s)
                        if pat == _
                        for p in ps
                    )
                    / pattern_counter[pat]
                )
            }"
        )
        param_len = (
            sum(
                p.bit_length()
                for s in pattern_matches[pat]
                for _, ps in extract_template(s)
                if pat == _
                for p in ps
            )
            / pattern_counter[pat]
        )
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        print(f"symb len: {symb_len}")
        print(f"pattern len: {pat.bit_length()}")

    # Might need
    # and pattern_counter[node] > tree_count (lattice_count)
    # You may want more than 1 per full tree ?
    common_patterns = [pat for pat in common_patterns if bit_gain(pat) > 0]

    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]


def symbolize_pattern(
    trees: Sequence[KNode[T]],
    symbols: Sequence[KNode[T]],
    new_symbol: KNode[T],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    index = IndexValue(len(symbols))
    trees = tuple(
        node_to_symbolized_node(index, new_symbol, tree) for tree in trees
    )
    symbols = tuple(
        node_to_symbolized_node(index, new_symbol, tree) for tree in symbols
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
    i = 0
    # Phase 1: Non-symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not is_symbolized(c) and not isinstance(c, RootNode)
        ]
        # print(f"candidates len: {len(candidates)}")
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])
        i += 1

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
        i += 1

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
            return SymbolNode(new_index, node.parameters)
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
            return SymbolNode(new_index, node.parameters)
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


def unsymbolize(knode: KNode[T], symbol_table: Sequence[KNode[T]]) -> KNode[T]:
    """
    Completely unsymbolize a given node.
    If the symbol table contains all the referenced templates, the resulting node
    should be free of any SymbolNodes or NestedNodes
    Second hypothesis: the symbol table has been completely resolved too.

    Args:
        knode: KNode containing NestedNodes and SymbolNodes.
        symbol_table: Symbol table containing all the templates referenced by NestedNodes and SymbolNodes
    """
    nnode = knode
    # Step 1: First unsymbolize SymbolNode
    nnode = resolve_symbols(nnode, symbol_table)

    # Step 2: Then unsymbolize NestedNodes
    nnode = expand_all_nested_nodes(nnode, symbol_table)
    return nnode


def unsymbolize_all(
    trees: Sequence[KNode[T]], symbol_table: Sequence[KNode[T]]
) -> tuple[KNode[T], ...]:
    """
    Hypothesis: no loop in the symbol table, a symbol can only reference a symbol strictly after him
    """
    # Step 1: Resolve the symbol table
    symbol_table = tuple(symbol_table)
    nsymbol_table = []
    for i, symb in enumerate(symbol_table):
        nsymbol_table.append(resolve_symbols(symb, symbol_table))

    nsymbol_table = tuple(nsymbol_table)

    # Step 2: Unsymbolize all the nodes
    return tuple(unsymbolize(tree, nsymbol_table) for tree in trees)


def full_symbolization(
    trees: Sequence[KNode[T]],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Full standard symbolization + node nesting.
    """
    # Step 1: nest nodes
    symbol_table = []
    nested = tuple(
        extract_nested_patterns(symbol_table, syntax_tree)
        for syntax_tree in trees
    )

    symbolized, symbol_table = symbolize(tuple(nested), tuple(symbol_table))

    return symbolized, symbol_table


# Helpers for tests
def root_to_symbolize():
    # Step 1: Define Primitive Nodes
    move0 = PrimitiveNode(MoveValue(0))
    move1 = PrimitiveNode(MoveValue(1))
    move2 = PrimitiveNode(MoveValue(2))
    move3 = PrimitiveNode(MoveValue(3))
    move5 = PrimitiveNode(MoveValue(5))
    move6 = PrimitiveNode(MoveValue(6))
    move7 = PrimitiveNode(MoveValue(7))

    # Step 2: Define Repeat Nodes
    repeat0 = RepeatNode(move0, CountValue(4))
    repeat1 = RepeatNode(move1, CountValue(4))
    repeat2 = RepeatNode(move2, CountValue(4))
    repeat3 = RepeatNode(move3, CountValue(4))

    # Step 3: Build the Nested Structure
    level3 = ProductNode((move7, repeat3))  # "7(3)*{4}"
    level2 = SumNode(frozenset({repeat2, level3}))  # "[(2)*{4}|7(3)*{4}]"
    level1 = ProductNode((move6, level2))  # "6[(2)*{4}|7(3)*{4}]"
    level0 = SumNode(
        frozenset({repeat1, level1})
    )  # "[(1)*{4}|6[(2)*{4}|7(3)*{4}]]"
    level_1 = ProductNode((move5, level0))  # "5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]"
    level_2 = SumNode(
        frozenset({repeat0, level_1})
    )  # "[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]]"
    program = ProductNode(
        (move0, level_2)
    )  # "0[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]]"

    # Step 4: Create the Root Node
    root_node = RootNode(
        program, CoordValue(Coord(5, 5)), PaletteValue(frozenset({1}))
    )

    # Verify the string representation
    # print(
    #     str(root_node)
    # )  # Should output: "Root(0[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]], (5, 5), {1})"

    return root_node


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
    assert result == node, (
        "Test Case 2 Failed: Single node should not be wrapped in ProductNode"
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
    expected = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3))
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
    expected = RepeatNode(PrimitiveNode(MoveValue(5)), CountValue(10))
    assert result == expected, (
        "Test Case 8 Failed: All identical nodes should compress into one RepeatNode"
    )

    # Test Case 9: Input as Iterator
    nodes = iter([PrimitiveNode(MoveValue(1))] * 4)
    result = encode_run_length(nodes)
    expected = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(4))
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
    root = RootNode(
        program, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
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
    assert shifted_root.position == CoordValue(Coord(0, 0)), (
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

    # Test Case 10: Sequence of RepeatNodes with same count
    r0 = RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(5))
    r1 = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(5))
    r2 = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(5))
    r3 = RepeatNode(PrimitiveNode(MoveValue(3)), CountValue(5))
    nodes = [r0, r1, r2, r3]
    result = get_iterator(nodes)
    assert len(result) == 1, (
        "Test Case 10 Failed: Should return a frozenset with one RepeatNode"
    )
    repeat_node = next(iter(result))
    assert isinstance(repeat_node, RepeatNode), (
        "Test Case 10 Failed: Result should be a RepeatNode"
    )
    assert repeat_node.node in nodes, (
        "Test Case 10 Failed: The node should be one of the original nodes"
    )
    assert isinstance(repeat_node.count, CountValue) and abs(
        repeat_node.count.value
    ) == len(nodes), (
        "Test Case 10 Failed: The count should equal the number of nodes"
    )
    # Verify that shifts regenerate the original set
    if repeat_node.count.value > 0:
        shifts = range(repeat_node.count.value)
    else:
        shifts = range(0, repeat_node.count.value, -1)
    expected = {shift(repeat_node.node, k) for k in shifts}
    assert expected == frozenset(nodes), (
        "Test Case 10 Failed: Shifts should regenerate the original set"
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
    expected = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3))
    assert result == expected, (
        f"Test Case 3 Failed: Repeating nodes should be compressed. Got {[result]} instead of {[expected]}"
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
    expected = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(0)))),
        CountValue(-3),
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
        var_child, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
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


def test_arity():
    """
    Test function for the `arity` function, which computes the number of parameters
    in a Kolmogorov Tree pattern based on the highest variable index plus one.
    """
    # Test Case 1: Single variable with index 0
    node1 = VariableNode(VariableValue(0))
    assert arity(node1) == 1, (
        "Test Case 1 Failed: Expected arity 1 for VariableNode(0)"
    )

    # Test Case 2: No variables
    node2 = PrimitiveNode(MoveValue(2))
    assert arity(node2) == 0, (
        "Test Case 2 Failed: Expected arity 0 for no variables"
    )

    # Test Case 3: Two variables with indices 0 and 1
    node3 = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(1)))
    )
    assert arity(node3) == 2, (
        f"Test Case 3 Failed: Expected arity 2 for indices [0, 1], got {arity(node3)}"
    )

    # Test Case 4: Single variable in a RepeatNode
    node4 = RepeatNode(VariableNode(VariableValue(0)), CountValue(4))
    assert arity(node4) == 1, (
        "Test Case 4 Failed: Expected arity 1 for index [0]"
    )

    # Test Case 5: Variables with indices 0 and 2 in a SumNode
    node5 = SumNode(
        frozenset(
            {VariableNode(VariableValue(0)), VariableNode(VariableValue(2))}
        )
    )
    assert arity(node5) == 3, (
        "Test Case 5 Failed: Expected arity 3 for indices [0, 2]"
    )

    # Test Case 6: Variable in a NestedNode
    node6 = NestedNode(
        IndexValue(0), VariableNode(VariableValue(1)), CountValue(3)
    )
    assert arity(node6) == 2, (
        "Test Case 6 Failed: Expected arity 2 for index [1]"
    )

    # Test Case 7: Variable in a SymbolNode's parameters
    node7 = SymbolNode(IndexValue(0), (VariableNode(VariableValue(0)),))
    assert arity(node7) == 1, (
        "Test Case 7 Failed: Expected arity 1 for index [0]"
    )

    # Test Case 8: No variables in a RepeatNode
    node8 = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(4))
    assert arity(node8) == 0, (
        "Test Case 8 Failed: Expected arity 0 for no variables"
    )

    # Test Case 9: Variables in nested structure
    node9 = ProductNode(
        (
            VariableNode(VariableValue(0)),
            RepeatNode(VariableNode(VariableValue(1)), CountValue(3)),
        )
    )
    assert arity(node9) == 2, (
        "Test Case 9 Failed: Expected arity 2 for indices [0, 1]"
    )

    # Test Case 10: Same variable used multiple times
    node10 = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(0)))
    )
    assert arity(node10) == 1, (
        "Test Case 10 Failed: Expected arity 1 for index [0]"
    )

    print("All arity tests passed successfully!")


def test_extract_nested_sum_template():
    """Tests the extract_nested_sum_template function."""

    def create_move_node(value):
        return PrimitiveNode(MoveValue(value))

    a, b, c, d, e, f, g, h = [create_move_node(i) for i in range(8)]
    inner_sum1 = SumNode(frozenset([b, c]))
    inner_sum2 = SumNode(frozenset([f, g]))
    prod1 = ProductNode((a, inner_sum1, d))
    prod2 = ProductNode((e, inner_sum2, h))
    main_sum = SumNode(frozenset([prod1, prod2]))
    result = extract_nested_sum_template(main_sum)
    assert result is not None, "Expected a template and parameter"
    template, parameter = result
    var_node = VariableNode(VariableValue(0))
    expected_template1 = SumNode(
        frozenset([ProductNode((a, var_node, d)), prod2])
    )
    expected_template2 = SumNode(
        frozenset([prod1, ProductNode((e, var_node, h))])
    )
    if template.children == expected_template1.children:
        assert parameter == inner_sum1, "Parameter should be inner_sum1"
    elif template.children == expected_template2.children:
        assert parameter == inner_sum2, "Parameter should be inner_sum2"
    else:
        assert False, f"Unexpected template structure: {str(template)}"
    sum_no_product = SumNode(frozenset([a, b, c]))
    assert extract_nested_sum_template(sum_no_product) is None, "Expected None"
    prod_no_sum = ProductNode((a, b, c))
    sum_with_prod_no_sum = SumNode(frozenset([prod_no_sum]))
    assert extract_nested_sum_template(sum_with_prod_no_sum) is None, (
        "Expected None"
    )
    print("Test extract_nested_sum_template - Passed")


def test_extract_nested_product_template():
    """Tests the extract_nested_product_template function."""

    def create_move_node(value):
        return PrimitiveNode(MoveValue(value))

    a, b, c, d, e, f = [create_move_node(i) for i in range(6)]
    inner_prod1 = ProductNode((b, c))
    inner_prod2 = ProductNode((d, e))
    sum_inner = SumNode(frozenset([inner_prod1, inner_prod2]))
    main_product = ProductNode((a, sum_inner, f))
    result = extract_nested_product_template(main_product)
    assert result is not None, "Expected a template and parameter"
    template, parameter = result
    var_node = VariableNode(VariableValue(0))
    expected_template1 = ProductNode(
        (a, SumNode(frozenset([var_node, inner_prod2])), f)
    )
    expected_template2 = ProductNode(
        (a, SumNode(frozenset([inner_prod1, var_node])), f)
    )
    if template.children == expected_template1.children:
        assert parameter == inner_prod1, "Parameter should be inner_prod1"
    elif template.children == expected_template2.children:
        assert parameter == inner_prod2, "Parameter should be inner_prod2"
    else:
        assert False, f"Unexpected template structure: {str(template)}"
    product_no_sum = ProductNode((a, b, c))
    assert extract_nested_product_template(product_no_sum) is None, (
        "Expected None"
    )
    sum_no_product = SumNode(frozenset([a, b]))
    product_with_sum = ProductNode((sum_no_product, c))
    assert extract_nested_product_template(product_with_sum) is None, (
        "Expected None"
    )
    print("Test extract_nested_product_template - Passed")


def test_node_to_symbolize():
    node = SumNode(
        frozenset(
            {
                SymbolNode(IndexValue(4), ()),
                RootNode(
                    SymbolNode(IndexValue(0), ()),
                    CoordValue(Coord(4, 4)),
                    PaletteValue(frozenset({3})),
                ),
                SymbolNode(IndexValue(6), ()),
                SymbolNode(
                    IndexValue(3),
                    (CoordValue(Coord(6, 0)),),
                ),
                SymbolNode(IndexValue(5), ()),
            }
        )
    )
    pattern = SumNode(
        frozenset(
            {
                SymbolNode(IndexValue(4), ()),
                SymbolNode(IndexValue(6), ()),
                VariableNode(VariableValue(0)),
                SymbolNode(
                    IndexValue(3),
                    (CoordValue(Coord(6, 0)),),
                ),
                SymbolNode(
                    IndexValue(5),
                    (
                        # RootNode(
                        #     SymbolNode(IndexValue(0), ()),
                        #     CoordValue(Coord(4, 4)),
                        #     PaletteValue(frozenset({3})),
                        # ),
                    ),
                ),
            }
        )
    )

    symbolized = node_to_symbolized_node(IndexValue(6), pattern, node)
    s2 = abstract_node(IndexValue(6), pattern, node)
    print(f"s2: {s2}")

    expected = SymbolNode(
        IndexValue(6),
        (
            RootNode(
                SymbolNode(IndexValue(0), ()),
                CoordValue(Coord(4, 4)),
                PaletteValue(frozenset({3})),
            ),
        ),
    )

    assert symbolized == expected, f"{symbolized} != {expected}"


def test_nested_collection_to_nested_node():
    """
    Tests the `nested_collection_to_nested_node` function for capturing recursive patterns in SumNodes.
    Verifies template extraction, terminal node identification, recursion count, and reconstruction.
    """

    # Helper functions to expand NestedNode for testing
    def substitute(template, substitution):
        """Substitutes VariableNodes in a template with given nodes."""
        if isinstance(template, SumNode):
            new_children = frozenset(
                substitute(child, substitution) for child in template.children
            )
            return SumNode(new_children)
        elif isinstance(template, ProductNode):
            new_children = tuple(
                substitute(child, substitution) for child in template.children
            )
            return ProductNode(new_children)
        elif isinstance(template, VariableNode):
            return substitution[template.index.value]
        return template

    def expand_nested(template, terminal, count):
        """Expands a NestedNode by applying the template `count` times starting from the terminal."""
        current = terminal
        for _ in range(count):
            current = substitute(template, {0: current})
        return current

    # Test Case 1: Recursive SumNode with multiple levels
    # Pattern: [0 Rec | 4], repeated 3 times for simplicity (scalable to 8 as in your example)
    terminal = SumNode(
        frozenset([PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))])
    )
    level1 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), terminal)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )
    level2 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), level1)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )
    level3 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), level2)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )

    result = nested_collection_to_nested_node(level3)
    assert result is not None, (
        "Expected a NestedNode and template for recursive SumNode"
    )
    nested_node, template = result

    # Expected template: [0 Var(0) | 4]
    var_node = VariableNode(VariableValue(0))
    expected_template = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), var_node)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )

    # Verify template
    assert template == expected_template, (
        f"Template mismatch: expected {expected_template}, got {template}"
    )

    # Verify NestedNode properties
    assert isinstance(nested_node, NestedNode), "Result should be a NestedNode"
    assert nested_node.node == terminal, (
        f"Terminal node mismatch: expected {terminal}, got {nested_node.node}"
    )
    assert isinstance(nested_node.count, CountValue)

    assert nested_node.count.value == 3, (
        f"Count mismatch: expected 3, got {nested_node.count.value}"
    )
    # Index is a placeholder (0), not critical for this test

    # Verify reconstruction
    expanded = expand_nested(
        template, nested_node.node, nested_node.count.value
    )
    assert expanded == level3, "Expanded node should match original level3"

    # Test Case 2: Non-recursive SumNode
    non_recursive = SumNode(
        frozenset([PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))])
    )
    result = nested_collection_to_nested_node(non_recursive)
    assert result is None, "Expected None for non-recursive SumNode"

    # Test Case 3: Recursive SumNode with single level
    result = nested_collection_to_nested_node(level1)
    assert result is not None, (
        "Expected a NestedNode for single-level recursive SumNode"
    )
    nested_node, template = result
    assert template == expected_template, (
        "Template mismatch in single-level case"
    )
    assert isinstance(nested_node.count, CountValue)
    assert nested_node.count.value == 1, (
        f"Count should be 1, got {nested_node.count.value}"
    )
    assert nested_node.node == terminal, (
        "Terminal node mismatch in single-level case"
    )
    expanded = expand_nested(
        template, nested_node.node, nested_node.count.value
    )
    assert expanded == level1, "Expanded node should match original level1"

    print("Test nested_collection_to_nested_node - Passed")


def test_symbolize_pattern():
    root_node = root_to_symbolize()

    # Create PrimitiveNodes
    move0 = PrimitiveNode(MoveValue(0))
    move5 = PrimitiveNode(MoveValue(5))
    move6 = PrimitiveNode(MoveValue(6))
    move7 = PrimitiveNode(MoveValue(7))

    # Create SymbolNodes
    s0_3 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(3)),))
    s0_2 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(2)),))
    s0_1 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(1)),))
    s0_0 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(0)),))

    # Build the tree bottom-up
    inner_sum = SumNode(frozenset({ProductNode((move7, s0_3)), s0_2}))
    prod6 = ProductNode((move6, inner_sum))
    sum2 = SumNode(frozenset({prod6, s0_1}))
    prod5 = ProductNode((move5, sum2))
    sum3 = SumNode(frozenset({prod5, s0_0}))
    program = ProductNode((move0, sum3))

    # Create the RootNode
    root_symbolized = RootNode(
        program, CoordValue(Coord(5, 5)), PaletteValue(frozenset({1}))
    )

    symbol = RepeatNode(VariableNode(VariableValue(0)), CountValue(4))
    r_symb, sym_table = symbolize_pattern((root_node,), tuple(), symbol)
    assert len(r_symb) == 1
    assert len(sym_table) == 1
    assert r_symb[0] == root_symbolized
    assert sym_table[0] == symbol


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
        f"Parameters should substitute into RepeatNode's count: {expected}, {result}"
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
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    tree3 = RootNode(
        pattern, CoordValue(Coord(2, 2)), PaletteValue(frozenset({3}))
    )
    tree4 = RootNode(
        pattern, CoordValue(Coord(3, 3)), PaletteValue(frozenset({4}))
    )
    trees = (tree1, tree2, tree3, tree4)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 2, (
        "Test Case 1 Failed: Should find exactly two candidate"
    )
    assert candidates[0] == RootNode(
        pattern, VariableNode(VariableValue(0)), VariableNode(VariableValue(1))
    ), "Test Case 1 Failed: Candidate should be the repeating pattern"
    print("Test Case 1: Basic Functionality - Passed")

    # Test Case 2: Frequency Threshold - Pattern Below Threshold
    unique_tree = RootNode(
        PrimitiveNode(MoveValue(3)),
        CoordValue(Coord(0, 0)),
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
        short_pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree6 = RootNode(
        short_pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
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
        CoordValue(Coord(0, 0)),
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
        CoordValue(Coord(1, 1)),
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
        CoordValue(Coord(2, 2)),
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
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    tree11 = RootNode(
        PrimitiveNode(MoveValue(2)),
        CoordValue(Coord(1, 1)),
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

    # Test Case 8: Integration with extract_template
    node = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    abstractions = extract_template(node)

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

def test_unify_sum():
    """
    Tests the unify function specifically for the SumNode case,
    focusing on correctness and deterministic binding.
    """

    # Test Case 1: Exact Match (No Variables), different frozenset order
    pattern1 = SumNode(frozenset({create_move_node(1), create_move_node(2), create_move_node(3)}))
    subtree1a = SumNode(frozenset({create_move_node(3), create_move_node(1), create_move_node(2)}))
    subtree1b = SumNode(frozenset({create_move_node(1), create_move_node(2), create_move_node(3)}))
    bindings1a: Bindings = {}
    bindings1b: Bindings = {}
    assert unify(pattern1, subtree1a, bindings1a)
    assert bindings1a == {}, "Exact match should produce empty bindings"
    assert unify(pattern1, subtree1b, bindings1b)
    assert bindings1b == {}, "Exact match should produce empty bindings regardless of order"

    # Test Case 2: Mismatch - Different Number of Children
    pattern2 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    subtree2 = SumNode(frozenset({create_move_node(1), create_move_node(2), create_move_node(3)}))
    bindings2: Bindings = {}
    assert not unify(pattern2, subtree2, bindings2), "Should fail on different child count"
    assert bindings2 == {}, "Bindings should be empty after failed unification"

    # Test Case 3: Mismatch - Same Number, Different Content
    pattern3 = SumNode(frozenset({create_move_node(1), create_move_node(4)})) # create_move_node(4) instead of create_move_node(2)
    subtree3 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    bindings3: Bindings = {}
    assert not unify(pattern3, subtree3, bindings3), "Should fail on different content"
    assert bindings3 == {}, "Bindings should be empty after failed unification"

    # Test Case 4: Single Variable - Successful Binding
    pattern4 = SumNode(frozenset({create_variable_node(0), create_move_node(2)})) # Var(0) and create_move_node(2)
    subtree4 = SumNode(frozenset({create_move_node(1), create_move_node(2)})) # Match create_move_node(2), Var(0) should bind to create_move_node(1)
    bindings4: Bindings = {}
    assert unify(pattern4, subtree4, bindings4), "Single variable unification should succeed"
    # Deterministic check: create_move_node(2) matches create_move_node(2). create_variable_node(0) must match the remaining create_move_node(1).
    assert bindings4 == {0: create_move_node(1)}, f"Expected Var(0) to bind to create_move_node(1), got {bindings4}"

    # Test Case 5: Single Variable - Binding to a Complex Node
    prod_node = ProductNode((create_move_node(3), create_move_node(4)))
    pattern5 = SumNode(frozenset({create_move_node(1), create_variable_node(0)}))
    subtree5 = SumNode(frozenset({create_move_node(1), prod_node}))
    bindings5: Bindings = {}
    assert unify(pattern5, subtree5, bindings5), "Variable binding to ProductNode should succeed"
    # create_move_node(1) matches create_move_node(1). create_variable_node(0) must bind to the remaining prod_node.
    assert bindings5 == {0: prod_node}, f"Expected Var(0) to bind to {prod_node}, got {bindings5}"

    # Test Case 6: Multiple Variables - Successful Binding & Determinism
    pattern6 = SumNode(frozenset({create_move_node(5), create_variable_node(1), create_variable_node(0)}))
    subtree6a = SumNode(frozenset({create_move_node(3), create_move_node(5), create_move_node(4)})) # Deliberate order difference
    subtree6b = SumNode(frozenset({create_move_node(4), create_move_node(3), create_move_node(5)})) # Another order
    bindings6a: Bindings = {}
    bindings6b: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted: ['5', 'Var(0)', 'Var(1)']
    # Subtree sorted: ['3', '4', '5']
    # Match '5' -> '5'.
    # Match 'Var(0)' -> '3'.
    # Match 'Var(1)' -> '4'.
    expected_bindings6 = {0: create_move_node(3), 1: create_move_node(4)}

    assert unify(pattern6, subtree6a, bindings6a), "Multi-variable unification (a) should succeed"
    assert bindings6a == expected_bindings6, f"Bindings (a) mismatch: Expected {expected_bindings6}, got {bindings6a}"

    assert unify(pattern6, subtree6b, bindings6b), "Multi-variable unification (b) should succeed"
    assert bindings6b == expected_bindings6, f"Bindings (b) mismatch: Expected {expected_bindings6}, got {bindings6b}"
    assert bindings6a == bindings6b, "Bindings should be deterministic regardless of initial frozenset order"

    # Test Case 7: Multiple Variables - Failure due to Content Mismatch
    pattern7 = SumNode(frozenset({create_variable_node(0), create_move_node(2)}))
    subtree7 = SumNode(frozenset({create_move_node(1), create_move_node(3)})) # No create_move_node(2) to match the concrete part
    bindings7: Bindings = {}
    assert not unify(pattern7, subtree7, bindings7), "Should fail if concrete parts don't match"
    assert bindings7 == {}, "Bindings should be empty after failed unification"

    # Test Case 8: Determinism Check with Swapped Variables in Pattern
    pattern8a = SumNode(frozenset({create_variable_node(0), create_variable_node(1)})) # Var(0), Var(1)
    pattern8b = SumNode(frozenset({create_variable_node(1), create_variable_node(0)})) # Var(1), Var(0) - same set
    subtree8 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    bindings8a: Bindings = {}
    bindings8b: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted (both cases): ['Var(0)', 'Var(1)']
    # Subtree sorted: ['1', '2']
    # Match 'Var(0)' -> '1'.
    # Match 'Var(1)' -> '2'.
    expected_bindings8 = {0: create_move_node(1), 1: create_move_node(2)}

    assert unify(pattern8a, subtree8, bindings8a)
    assert bindings8a == expected_bindings8, f"Bindings (8a) mismatch: Expected {expected_bindings8}, got {bindings8a}"
    # Even though pattern8b looks different, its frozenset is identical
    assert unify(pattern8b, subtree8, bindings8b)
    assert bindings8b == expected_bindings8, f"Bindings (8b) mismatch: Expected {expected_bindings8}, got {bindings8b}"

    # Test Case 9: Nested SumNode Unification
    inner_p9 = SumNode(frozenset({create_variable_node(1), create_move_node(6)}))
    inner_s9 = SumNode(frozenset({create_move_node(4), create_move_node(6)}))
    pattern9 = SumNode(frozenset({create_variable_node(0), inner_p9}))
    subtree9 = SumNode(frozenset({create_move_node(3), inner_s9}))
    bindings9: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted: [str(inner_p9), 'Var(0)'] -> ['[6|Var(1)]', 'Var(0)']
    # Subtree sorted: [str(create_move_node(3)), str(inner_s9)] -> ['3', '[4|6]']
    # Match '[6|Var(1)]' -> '[4|6]' (requires inner unification)
    #   Inner unification: p=['6', 'Var(1)'], s=['4', '6'] -> Match '6'->'6', Match 'Var(1)'->'4'. Bindings: {1: create_move_node(4)}
    # Match 'Var(0)' -> '3'. Bindings: {0: create_move_node(3)}
    # Combine bindings: {0: create_move_node(3), 1: create_move_node(4)}
    expected_bindings9 = {0: create_move_node(3), 1: create_move_node(4)}

    assert unify(pattern9, subtree9, bindings9), "Nested SumNode unification should succeed"
    assert bindings9 == expected_bindings9, f"Bindings (9) mismatch: Expected {expected_bindings9}, got {bindings9}"

    # Test Case 10: Variable bound to different type
    pattern10 = SumNode(frozenset({create_variable_node(0), create_move_node(1)}))
    subtree10 = SumNode(frozenset({RepeatNode(create_variable_node(0), CountValue(4)), create_move_node(1)})) # Bind Var(0) to a RepeatNode
    bindings10: Bindings = {}
    assert unify(pattern10, subtree10, bindings10), "Variable binding to CoordValue should succeed"
    assert bindings10 == {0: RepeatNode(create_variable_node(0), CountValue(4))}, f"Expected Var(0) to bind to {RepeatNode(create_variable_node(0), CountValue(4))}, got {bindings10}"

    print("\nAll test_unify_sum_deterministic tests passed!")


def test_expand_nested_node():
    """
    Tests the expand_nested_node function, which expands a NestedNode by recursively
    applying a template from a symbol table to a terminal node for a specified count.
    """
    # Common node definitions
    move0 = PrimitiveNode(MoveValue(0))  # Template prefix
    move1 = PrimitiveNode(MoveValue(1))  # Terminal node
    move2 = PrimitiveNode(MoveValue(2))  # SumNode alternative
    move3 = PrimitiveNode(MoveValue(3))  # Template suffix
    var0 = VariableNode(VariableValue(0))  # Variable for substitution

    # Test Case 1: Simple ProductNode Template with SumNode
    # Template: ProductNode((move3, SumNode(frozenset({move0, var0}))))
    template1 = ProductNode((move3, SumNode(frozenset({move0, var0}))))
    symbols1 = (template1,)

    # Count=1: ProductNode((move3, SumNode(frozenset({move0, move2}))))
    nested1 = NestedNode(IndexValue(0), move2, CountValue(1))
    expected1 = ProductNode((move3, SumNode(frozenset({move0, move2}))))
    result1 = expand_nested_node(nested1, symbols1)
    assert result1 == expected1, (
        f"Test Case 1 (count=1) Failed: Expected {expected1}, got {result1}"
    )

    # Count=2: ProductNode((move3, SumNode(frozenset({move0, expected1}))))
    nested2 = NestedNode(IndexValue(0), move2, CountValue(2))
    expected2 = ProductNode((move3, SumNode(frozenset({move0, expected1}))))
    result2 = expand_nested_node(nested2, symbols1)
    assert result2 == expected2, (
        f"Test Case 1 (count=2) Failed: Expected {expected2}, got {result2}"
    )

    # Count=3:
    nested3 = NestedNode(IndexValue(0), move2, CountValue(3))
    expected3 = ProductNode((move3, SumNode(frozenset({move0, expected2}))))
    result3 = expand_nested_node(nested3, symbols1)
    assert result3 == expected3, (
        f"Test Case 1 (count=3) Failed: Expected {expected3}, got {result3}"
    )

    # Test Case 2: Complex Template with SumNode
    # Template: ProductNode((MoveValue(0), SumNode({Var(0), MoveValue(2)}), MoveValue(3)))
    sum_template = SumNode(frozenset({var0, move3}))
    template2 = ProductNode((move0, sum_template, move3))
    symbols2 = (template2,)

    # Count=1: ProductNode((MoveValue(0), SumNode({MoveValue(1), MoveValue(3)}), MoveValue(3)))
    nested1_2 = NestedNode(IndexValue(0), move1, CountValue(1))
    expected1_2 = ProductNode(
        (move0, SumNode(frozenset({move1, move3})), move3)
    )
    result1_2 = expand_nested_node(nested1_2, symbols2)
    assert result1_2 == expected1_2, (
        f"Test Case 2 (count=1) Failed: Expected {expected1_2}, got {result1_2}"
    )

    # Count=2: ProductNode((MoveValue(0), SumNode({ProductNode((MoveValue(0), SumNode({MoveValue(1), MoveValue(3)}), MoveValue(3))), MoveValue(2)}), MoveValue(3)))
    nested2_2 = NestedNode(IndexValue(0), move1, CountValue(2))
    inner_sum = SumNode(frozenset({move1, move3}))
    inner_product = ProductNode((move0, inner_sum, move3))
    expected2_2 = ProductNode(
        (move0, SumNode(frozenset({inner_product, move3})), move3)
    )
    result2_2 = expand_nested_node(nested2_2, symbols2)
    assert result2_2 == expected2_2, (
        f"Test Case 2 (count=2) Failed: Expected {expected2_2}, got {result2_2}"
    )

    print("Test expand_nested_node - Passed")


def test_factor_by_existing_symbols():
    """Tests the factor_by_existing_symbols function for replacing matching patterns with SymbolNodes."""
    # Test Case 1: Basic Pattern Replacement
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    tree = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    symbols = (pattern,)
    result = factor_by_existing_symbols(tree, symbols)
    expected = RootNode(
        SymbolNode(IndexValue(0), ()),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert result == expected, (
        "Test Case 1 Failed: Should replace pattern with SymbolNode"
    )
    print("Test Case 1: Basic Pattern Replacement - Passed")

    # Test Case 2: No Matching Pattern
    tree = RootNode(
        PrimitiveNode(MoveValue(1)),
        CoordValue(Coord(0, 0)),
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
            SymbolNode(IndexValue(0), ()),
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
    symbol_node = SymbolNode(IndexValue(0), ())
    expected = RepeatNode(symbol_node, CountValue(2))
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
    expected = SymbolNode(IndexValue(1), (PrimitiveNode(MoveValue(1)),))
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
    print("Test Case 4: Merging with Nested Symbols and Variables - Passed")

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
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    trees = (tree1, tree2)
    symbol_tables = [(), ()]
    final_trees, final_symbols = symbolize_together(trees, symbol_tables)
    expected_symbol = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    expected_trees = (
        RootNode(
            SymbolNode(IndexValue(0), tuple()),
            CoordValue(Coord(0, 0)),
            PaletteValue(value=frozenset({1})),
        ),
        RootNode(
            SymbolNode(IndexValue(0), tuple()),
            CoordValue(Coord(1, 1)),
            PaletteValue(value=frozenset({2})),
        ),
    )
    expected_symbols = (expected_symbol,)
    assert final_trees == expected_trees, (
        f"Test Case 1 Failed: Trees incorrect, got {final_trees}"
    )
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
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
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
    assert (
        move_right.bit_length()
        == BitLength.NODE_TYPE + MoveValue(2).bit_length()
    ), (
        f"PrimitiveNode bit length should be {BitLength.NODE_TYPE + MoveValue(2).bit_length()} ( {BitLength.NODE_TYPE} + {MoveValue(2).bit_length()})"
    )
    product = ProductNode((move_right, move_right))
    assert (
        product.bit_length()
        == BitLength.NODE_TYPE + 2 + 2 * move_right.bit_length()
    ), (
        f"ProductNode bit length should be {BitLength.NODE_TYPE + 2 + 2 * move_right.bit_length()} ({BitLength.NODE_TYPE} + 2 + 2 * {move_right.bit_length()})"
    )
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode(frozenset((move_right, move_down)))
    assert (
        sum_node.bit_length()
        == BitLength.NODE_TYPE
        + 2
        + move_right.bit_length()
        + move_down.bit_length()
    ), (
        f"SumNode bit length should be {BitLength.NODE_TYPE + 2 + move_right.bit_length() + move_down.bit_length()} ({BitLength.NODE_TYPE} + 2 + {move_right.bit_length()} + {move_down.bit_length()})"
    )
    repeat = RepeatNode(move_right, CountValue(3))
    assert (
        repeat.bit_length()
        == BitLength.NODE_TYPE + BitLength.COUNT + move_right.bit_length()
    ), (
        f"RepeatNode bit length should be {BitLength.NODE_TYPE + BitLength.COUNT + move_right.bit_length()} ({BitLength.NODE_TYPE} + {move_right.bit_length()} + {BitLength.COUNT})"
    )
    nested = NestedNode(IndexValue(0), repeat, CountValue(3))
    assert (
        nested.bit_length()
        == BitLength.NODE_TYPE
        + BitLength.COUNT
        + BitLength.INDEX
        + repeat.bit_length()
    ), f"RepeatNode bit length should be {
        BitLength.NODE_TYPE
        + BitLength.INDEX
        + BitLength.COUNT
        + sum_node.bit_length()
        + repeat.bit_length()
    } ({BitLength.NODE_TYPE} + {BitLength.INDEX} + {repeat.bit_length()} + {
        BitLength.COUNT
    })"
    symbol = SymbolNode(IndexValue(0), ())
    assert symbol.bit_length() == BitLength.NODE_TYPE + BitLength.INDEX, (
        f"SymbolNode bit length should be {BitLength.NODE_TYPE + BitLength.INDEX} ({BitLength.NODE_TYPE} + {BitLength.INDEX})"
    )
    root = RootNode(
        product, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    assert (
        root.bit_length()
        == BitLength.NODE_TYPE
        + product.bit_length()
        + ARCBitLength.COORD
        + ARCBitLength.COLORS
    ), (
        "RootNode bit length should be {BitLength.NODE_TYPE + product.bit_length() + ARCBitLength.COORD + ARCBitLength.COLORS} ({BitLength.NODE_TYPE} + {product.bit_length()} + {ARCBitLength.COORD} + {ARCBitLength.COLORS})"
    )
    print("Test 1: Basic node creation and bit length - Passed")

    # Test 2: String representations
    assert str(move_right) == "2", "PrimitiveNode str should be '2'"
    assert str(product) == "22", "ProductNode str should be '22'"
    assert str(sum_node) == "[2|3]", "SumNode str should be '[2|3]'"
    assert str(repeat) == "(2)*{3}", "RepeatNode str should be '(2)*{3}'"
    assert str(symbol).strip() == "s_0", (
        f"SymbolNode str should be 's_0', got {str(symbol)}"
    )
    assert str(root) == "Root(22, (0, 0), {1})", (
        f"RootNode str should be 'Root(22, (0, 0), {1})'. Got {str(root)} instead"
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
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    resolved = resolve_symbols(root_with_symbol, symbols)
    expected_resolved = RootNode(
        ProductNode((move_right, move_down, move_right)),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

    test_encode_run_length()
    test_construct_product_node()
    test_shift()
    test_get_iterator()
    test_arity()

    test_find_repeating_pattern()
    test_factorize_tuple()
    test_is_abstraction()
    test_resolve_symbols()
    test_find_symbol_candidates()
    test_matching()

    test_unify_sum()

    test_extract_nested_sum_template()
    test_node_to_symbolize()
    test_extract_nested_product_template()
    test_nested_collection_to_nested_node()
    test_expand_nested_node()
    test_factor_by_existing_symbols()
    test_symbolize_pattern()
    test_remap_symbol_indices()
    test_remap_sub_symbols()
    test_merge_symbol_tables()
    test_symbolize_together()


if __name__ == "__main__":
    run_tests()
    print("All tests passed successfully!")
