"""
Node types for the Kolmogorov Tree.

This module defines the AST node types:
- KNode: Base class for all nodes
- PrimitiveNode: Leaf node holding a primitive value
- VariableNode: Variable placeholder within a symbol
- CollectionNode: Abstract base for nodes with children
- ProductNode: Sequence of actions (AND)
- SumNode: Choice among alternatives (OR)
- RepeatNode: Repetition of a node
- NestedNode: Fixed point combinator for recursive patterns
- SymbolNode: Abstraction/reusable pattern reference
- RectNode: ARC-specific rectangle node
- RootNode: Entry point with position and colors
"""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Generic

from localtypes import BitLengthAware

from kolmogorov_tree.primitives import (
    BitLength,
    CoordValue,
    CountValue,
    IndexValue,
    NoneValue,
    PaletteValue,
    T,
    VariableValue,
)


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

    def __or__(self, other: KNode[T]) -> SumNode[T]:
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

    def __and__(self, other: KNode[T]) -> ProductNode[T]:
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

    def __add__(self, other: KNode[T]) -> ProductNode[T]:
        """Overloads + for concatenation, unpacking ProductNodes."""
        return self.__and__(other)  # Same behavior as &

    def __mul__(self, count: int) -> RepeatNode[T]:
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
        return f"[{','.join(str(child) for child in self.children)}]"


@dataclass(frozen=True)
class SumNode(CollectionNode[T]):
    """Represents a choice among alternatives (OR operation)."""

    children: frozenset[KNode[T]] = field(default_factory=frozenset)

    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        # Sort children by string representation for consistent output
        children_str = [str(child) for child in self.children]  # Compute once
        sorted_str = sorted(children_str)  # Sort precomputed strings
        return "{" + ",".join(sorted_str) + "}"  # Use precomputed strings


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

    Note: Use `resolve()` from `kolmogorov_tree.resolution` to expand this node.
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
    """
    Represents an abstraction or reusable pattern.

    Note: Use `resolve()` from `kolmogorov_tree.resolution` to expand this node.
    """

    index: IndexValue  # Index in the symbol table
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_len

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}()"


# Dirty hack for ARC
# It will need to be an abstraction
# That will be pre-populated in the symbol table
# akin to pre-training
@dataclass(frozen=True)
class RectNode(KNode):
    height: CountValue | VariableNode
    width: CountValue | VariableNode

    def bit_length(self) -> int:
        # 3 bits for node type + 8 bits for ARC-specific rectangle encoding
        height_len = (
            BitLength.COUNT
            if isinstance(self.height, int)
            else self.height.bit_length()
        )
        width_len = (
            BitLength.COUNT if isinstance(self.width, int) else self.width.bit_length()
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
        return super().bit_length() + self.node.bit_length() + pos_len + colors_len

    def __str__(self) -> str:
        position = self.position
        pos_str = str(position)
        colors_str = str(self.colors)
        node_str = str(self.node)
        return f"Root({node_str}, {pos_str}, {colors_str})"


# Type aliases
Unsymbolized = PrimitiveNode | RepeatNode | RootNode | ProductNode | SumNode | RectNode
Uncompressed = PrimitiveNode | ProductNode | SumNode

__all__ = [
    "KNode",
    "PrimitiveNode",
    "VariableNode",
    "CollectionNode",
    "ProductNode",
    "SumNode",
    "RepeatNode",
    "NestedNode",
    "SymbolNode",
    "RectNode",
    "RootNode",
    "Unsymbolized",
    "Uncompressed",
]
