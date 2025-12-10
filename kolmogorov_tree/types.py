"""
Core type abstractions for Algorithmic Information Theory.

These types form the foundation for MDL (Minimum Description Length) based
tree representations, independent of any specific domain like ARC.

Types:
    BitLengthAware  - Base interface for objects with measurable description length
    Primitive       - Wrapper for leaf values in syntax trees
    KeyValue        - Position identifiers in tree structures
    Resolvable      - Protocol for symbol-table-resolvable references
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable


@dataclass(frozen=True, slots=True)
class BitLengthAware(ABC):
    """
    Base interface for values that contribute to description length.

    In MDL/AIT, the "cost" of an object is measured in bits. This interface
    ensures all tree components can report their encoding cost.
    """

    @abstractmethod
    def bit_length(self) -> int:
        """Returns the encoding cost in bits."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable representation."""
        ...


@dataclass(frozen=True, slots=True)
class Primitive(BitLengthAware):
    """
    Wrapper for primitive/leaf values in syntax trees.

    Subclasses define specific value types (counts, indices, domain values)
    and their bit-length encoding costs.
    """

    value: Any

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class KeyValue(Primitive):
    """
    Position identifier in tree structures.

    Used to locate nodes during tree traversal and edit operations.
    The bit_length is 0 because MDL costs for keys are computed externally
    based on context (sequence length, field count, etc.).

    Value types:
        - str: Named field in a dataclass
        - None: Primitive value or frozenset element
        - tuple[str | None, int]: Indexed position in a sequence
    """

    value: str | None | tuple[str | None, int]

    def bit_length(self) -> int:
        return 0


# Type variable for generic node operations
TNode = TypeVar("TNode", bound=BitLengthAware)


@runtime_checkable
class Resolvable(Protocol[TNode]):
    """
    Protocol for references that can be resolved via a symbol table.

    Used by SymbolNode and NestedNode to support compression through
    shared substructure references.
    """

    def resolve(self, symbol_table: Sequence[TNode]) -> TNode:
        """Dereference using the symbol table."""
        ...

    def eq_ref(self, other: "Resolvable[TNode]") -> bool:
        """Check if two references point to the same symbol."""
        ...


__all__ = [
    "BitLengthAware",
    "Primitive",
    "KeyValue",
    "TNode",
    "Resolvable",
]
