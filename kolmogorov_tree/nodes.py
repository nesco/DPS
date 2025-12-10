"""
Node types for the Kolmogorov Tree AST.

Hierarchy:
    KNode (abstract)
    ├── PrimitiveNode      - Leaf holding a primitive value
    ├── VariableNode       - Lambda variable placeholder
    ├── CollectionNode (abstract)
    │   ├── ProductNode    - Ordered sequence (AND)
    │   └── SumNode        - Unordered alternatives (OR)
    ├── RepeatNode         - Repetition with count
    ├── NestedNode         - Fixed-point combinator for recursion
    ├── SymbolNode         - Reference to symbol table entry
    ├── RectNode           - ARC-specific rectangle
    └── RootNode           - Entry point with position/colors
"""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Generic

from kolmogorov_tree.types import BitLengthAware

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


@dataclass(frozen=True)
class KNode(Generic[T], BitLengthAware, ABC):
    """Base class for all Kolmogorov tree nodes."""

    def __len__(self) -> int:
        return self.bit_length()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def bit_length(self) -> int:
        return BitLength.NODE_TYPE

    def __or__(self, other: KNode[T]) -> SumNode[T]:
        """Creates SumNode: `a | b` means choice between a and b."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        left = list(self.children) if isinstance(self, SumNode) else [self]
        right = list(other.children) if isinstance(other, SumNode) else [other]
        return SumNode(frozenset(left + right))

    def __and__(self, other: KNode[T]) -> ProductNode[T]:
        """Creates ProductNode: `a & b` means sequence of a then b."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        left = list(self.children) if isinstance(self, ProductNode) else [self]
        right = list(other.children) if isinstance(other, ProductNode) else [other]
        return ProductNode(tuple(left + right))

    def __add__(self, other: KNode[T]) -> ProductNode[T]:
        """Alias for `&`: `a + b` means sequence."""
        return self.__and__(other)

    def __mul__(self, count: int) -> RepeatNode[T]:
        """Creates RepeatNode: `a * 3` means repeat a three times."""
        if not isinstance(count, int):
            raise TypeError("Count must be an integer")
        if count < 0:
            raise ValueError("Count must be non-negative")
        if (
            isinstance(self, RepeatNode)
            and isinstance(self.count, CountValue)
            and self.count.value * count < 2**BitLength.COUNT
        ):
            return RepeatNode(self.node, CountValue(self.count.value * count))
        return RepeatNode(self, CountValue(count))


@dataclass(frozen=True)
class PrimitiveNode(KNode[T]):
    """Leaf node holding a single primitive value (e.g., MoveValue, CountValue)."""

    value: T

    @property
    def data(self) -> Any:
        """Unwraps the underlying raw value."""
        return self.value.value

    def bit_length(self) -> int:
        return super().bit_length() + self.value.bit_length()

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class VariableNode(KNode[T]):
    """Lambda variable placeholder, bound during symbol resolution."""

    index: VariableValue

    def bit_length(self) -> int:
        return super().bit_length() + self.index.bit_length()

    def __str__(self) -> str:
        return f"Var({self.index})"


@dataclass(frozen=True)
class CollectionNode(KNode[T], ABC):
    """Abstract base for nodes containing multiple children."""

    children: Collection[KNode[T]]

    def bit_length(self) -> int:
        child_count = len(self.children)
        count_bits = math.ceil(math.log2(child_count + 1)) if child_count else 0
        children_bits = sum(child.bit_length() for child in self.children)
        return super().bit_length() + children_bits + count_bits


@dataclass(frozen=True)
class ProductNode(CollectionNode[T]):
    """Ordered sequence of nodes executed in order (conjunction/AND)."""

    children: tuple[KNode[T], ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        return f"[{','.join(str(c) for c in self.children)}]"


@dataclass(frozen=True)
class SumNode(CollectionNode[T]):
    """Unordered set of alternative nodes (disjunction/OR)."""

    children: frozenset[KNode[T]] = field(default_factory=frozenset)

    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        return "{" + ",".join(sorted(str(c) for c in self.children)) + "}"


@dataclass(frozen=True)
class RepeatNode(KNode[T]):
    """Repetition of a node pattern a specified number of times."""

    node: KNode[T]
    count: CountValue | VariableNode

    def bit_length(self) -> int:
        return super().bit_length() + self.node.bit_length() + self.count.bit_length()

    def __str__(self) -> str:
        return f"({self.node})*{{{self.count}}}"


@dataclass(frozen=True)
class NestedNode(KNode[T]):
    """
    Finite fixed-point combinator for recursive symbol application.

    Semantics:
        Y_0(i, node) = node
        Y_n(i, node) = s_i(Y_{n-1}(i, node))

    Use `resolve()` from kolmogorov_tree.resolution to expand.
    """

    index: IndexValue
    node: KNode[T]
    count: CountValue | VariableNode

    def bit_length(self) -> int:
        return (
            super().bit_length()
            + self.index.bit_length()
            + self.node.bit_length()
            + self.count.bit_length()
        )

    def __str__(self) -> str:
        return f"Y_{{{self.count}}}({self.index}, {self.node})"


@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """
    Reference to a reusable pattern in the symbol table.

    Use `resolve()` from kolmogorov_tree.resolution to expand.
    """

    index: IndexValue
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        params_bits = sum(p.bit_length() for p in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_bits

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}()"


@dataclass(frozen=True)
class RectNode(KNode):
    """ARC-specific filled rectangle primitive."""

    height: CountValue | VariableNode
    width: CountValue | VariableNode

    def bit_length(self) -> int:
        height_bits = (
            BitLength.COUNT
            if isinstance(self.height, int)
            else self.height.bit_length()
        )
        width_bits = (
            BitLength.COUNT if isinstance(self.width, int) else self.width.bit_length()
        )
        return super().bit_length() + height_bits + width_bits

    def __str__(self) -> str:
        return f"Rect({self.height}, {self.width})"


@dataclass(frozen=True)
class RootNode(KNode[T]):
    """Entry point wrapping a program with its execution context (position, colors)."""

    node: KNode[T] | NoneValue
    position: CoordValue | VariableNode
    colors: PaletteValue | VariableNode

    def bit_length(self) -> int:
        return (
            super().bit_length()
            + self.node.bit_length()
            + self.position.bit_length()
            + self.colors.bit_length()
        )

    def __str__(self) -> str:
        return f"Root({self.node}, {self.position}, {self.colors})"


# Type aliases for node categories
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
