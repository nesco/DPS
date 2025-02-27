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

from dataclasses import dataclass, field
from typing import Generic, TypeVar, Union, Iterator, Optional, Callable
from abc import ABC, abstractmethod


class BitLength(IntEnum):
    COORD = 10  # Base length for node type (3 bits) because <= 8 types
    COLOR = 4
    NODE = 3
    MOVE = 3  # Assuming 8-connectivity 3bits per move
    COUNT = 4  # counts of repeats should be an int between 2 and 9 or -2 and -8 (4 bits) ideally
    INDEX = 3  # should not be more than 8 so 3 bits
    INDEX_VARIABLE = 1  # Variable can be 0 or 1 so 1 bit
    RECT = 8


@dataclass(frozen=True)
class BitLengthAware(ABC):
    """Protocol for classes that know their bit lengths"""

    @abstractmethod
    def __len__(self) -> int:
        """Return the bit length of this value"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class Moves(BitLengthAware):
    "Freeman chain code moves"

    moves: str

    def __len__(self) -> int:
        return BitLength.NODE + BitLength.MOVE * len(self.moves)

    def __str__(self) -> str:
        return self.moves

@dataclass(frozen=True)
