"""
Primitive value types for the Kolmogorov Tree.

Types:
    Generic:
        CountValue    - Repeat count (5 bits, 0-31)
        VariableValue - Lambda variable index (1 bit, 0-1)
        IndexValue    - Symbol table index (7 bits, 0-127)
        NoneValue     - Null placeholder

    ARC-specific:
        MoveValue     - 8-directional move (3 bits, 0-7)
        PaletteValue  - Color set
        CoordValue    - 2D grid coordinate

    Abstract:
        Alphabet      - Base for shiftable output types (bound to TypeVar T)
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Self, TypeVar

from localtypes import Color, Coord, Primitive


class BitLength(IntEnum):
    """Bit widths for generic tree components."""

    COUNT = 5
    NODE_TYPE = 4
    INDEX = 7
    VAR = 1
    NONE = 0


class ARCBitLength(IntEnum):
    """Bit widths for ARC-AGI specific components."""

    COORD = 10
    COLORS = 4
    DIRECTIONS = 3


@dataclass(frozen=True)
class Alphabet(Primitive):
    """Base class for shiftable program output types (e.g., moves, colors)."""

    @abstractmethod
    def shift(self, k: int) -> Self:
        """Returns value shifted by k steps (with wraparound)."""
        pass

    @staticmethod
    def size() -> int:
        """Returns the alphabet cardinality."""
        return 0


@dataclass(frozen=True)
class CountValue(Primitive):
    """Repetition count (5-bit unsigned integer, 0-31)."""

    value: int

    def bit_length(self) -> int:
        return BitLength.COUNT


@dataclass(frozen=True)
class VariableValue(Primitive):
    """Lambda variable index (1-bit, 0-1)."""

    value: int

    def bit_length(self) -> int:
        return BitLength.VAR


@dataclass(frozen=True)
class IndexValue(Primitive):
    """Symbol table index (7-bit, 0-127)."""

    value: int

    def bit_length(self) -> int:
        return BitLength.INDEX


@dataclass(frozen=True)
class NoneValue(Primitive):
    """Null/empty placeholder."""

    value: None = None

    def bit_length(self) -> int:
        return BitLength.NONE


@dataclass(frozen=True)
class MoveValue(Alphabet):
    """8-directional grid move (0-7, where 0=right, 2=down, etc.)."""

    value: int

    def bit_length(self) -> int:
        return ARCBitLength.DIRECTIONS

    def shift(self, k: int) -> "MoveValue":
        return MoveValue((self.value + k) % 8)

    @staticmethod
    def size() -> int:
        return 8


@dataclass(frozen=True)
class PaletteValue(Primitive):
    """Set of ARC colors (0-9)."""

    value: frozenset[Color]

    def bit_length(self) -> int:
        return ARCBitLength.COLORS * len(self.value)

    def __str__(self) -> str:
        return f"set{set(self.value)}"


@dataclass(frozen=True)
class CoordValue(Primitive):
    """2D grid coordinate (5 bits per axis, 0-31 each)."""

    value: Coord

    def bit_length(self) -> int:
        return ARCBitLength.COORD

    def __str__(self) -> str:
        return str(tuple(self.value))


T = TypeVar("T", bound=Alphabet)

__all__ = [
    "BitLength",
    "ARCBitLength",
    "Alphabet",
    "CountValue",
    "VariableValue",
    "IndexValue",
    "NoneValue",
    "MoveValue",
    "PaletteValue",
    "CoordValue",
    "T",
]
