"""
Primitive types for the Kolmogorov Tree.

This module defines:
- Bit length enums (BitLength, ARCBitLength)
- Value types (CountValue, VariableValue, IndexValue, NoneValue)
- ARC-specific types (MoveValue, PaletteValue, CoordValue)
- The Alphabet base class for shiftable types
- The T TypeVar bound to Alphabet
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Self, TypeVar

from localtypes import Color, Coord, Primitive


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


# ARC Specific primitives
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
        return f"set{set(self.value)}"


@dataclass(frozen=True)
class CoordValue(Primitive):
    """Represents a 2D coordinate pair."""

    value: Coord

    def bit_length(self) -> int:
        return ARCBitLength.COORD  # 10 bits (5 per coordinate)

    def __str__(self) -> str:
        return str(tuple(self.value))


T = TypeVar("T", bound=Alphabet)

# Re-export BitLengthAware for convenience
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
