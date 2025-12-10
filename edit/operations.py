"""
Operation dataclasses for edit distance transformations.

Four base operations: Identity, Add, Delete, Substitute.
Extended operations add: Prune, Graft, Inner, Resolve.
"""

from dataclasses import dataclass

from kolmogorov_tree.types import BitLengthAware, KeyValue


type Operation = Add | Delete | Identity | Substitute | Inner

type ExtendedOperation = (
    Identity | Add | Delete | Substitute | Prune | Graft | Inner | Resolve
)


@dataclass(frozen=True)
class Inner(BitLengthAware):
    """Container for nested operations on children fields."""

    key: KeyValue
    children: "frozenset[ExtendedOperation]"

    def bit_length(self) -> int:
        return sum(op.bit_length() for op in self.children)

    def __str__(self) -> str:
        children_str = ", ".join(str(c) for c in self.children)
        return f"Inner({children_str})"


@dataclass(frozen=True)
class Identity(BitLengthAware):
    """Represents an element that remains unchanged."""

    key: KeyValue

    def bit_length(self) -> int:
        return 0

    def __str__(self) -> str:
        return "Identity"


@dataclass(frozen=True)
class Add(BitLengthAware):
    """Represents an element that is added to a collection."""

    key: KeyValue
    value: BitLengthAware

    def bit_length(self) -> int:
        return self.value.bit_length()

    def __str__(self) -> str:
        key_str = f" at {self.key}" if self.key.value is not None else ""
        return f"Add({self.value}{key_str})"


@dataclass(frozen=True)
class Delete(BitLengthAware):
    """Represents an element that is deleted from a collection."""

    key: KeyValue
    value: BitLengthAware  # Needed for apply_transformation to know what to remove

    def bit_length(self) -> int:
        return self.value.bit_length()

    def __str__(self) -> str:
        key_str = f" at {self.key}" if self.key.value is not None else ""
        return f"Delete({self.value}{key_str})"


@dataclass(frozen=True)
class Substitute(BitLengthAware):
    """Represents an element substituted with another."""

    key: KeyValue
    after: BitLengthAware

    def bit_length(self) -> int:
        return self.after.bit_length()

    def __str__(self) -> str:
        key_str = f" at {self.key}" if self.key.value is not None else ""
        return f"Substitute({self.after}{key_str})"


@dataclass(frozen=True)
class Prune(BitLengthAware):
    """Keep only a descendant at the given key, removing all ancestors."""

    key: KeyValue
    then: ExtendedOperation

    def bit_length(self) -> int:
        return self.key.bit_length() + self.then.bit_length()

    def __str__(self) -> str:
        if isinstance(self.key.value, tuple):
            field, index = self.key.value
            return f"Prune({field}[{index}], {self.then})"
        return f"Prune({self.key}, {self.then})"


@dataclass(frozen=True)
class Graft(BitLengthAware):
    """Attach source as a descendant of a new ancestor structure."""

    key: KeyValue
    ancestor: BitLengthAware
    then: ExtendedOperation

    def bit_length(self) -> int:
        return (
            self.key.bit_length() + self.ancestor.bit_length() + self.then.bit_length()
        )

    def __str__(self) -> str:
        key_str = self.key.value if self.key.value is not None else "root"
        return f"Graft({self.ancestor}[{key_str}], {self.then})"


@dataclass(frozen=True)
class Resolve(BitLengthAware):
    """Resolve a SymbolNode reference to its concrete subtree."""

    key: KeyValue
    then: ExtendedOperation

    def bit_length(self) -> int:
        return self.then.bit_length()

    def __str__(self) -> str:
        return f"Resolve({self.then})"


def identity_or_inner(
    distance: int,
    operation: ExtendedOperation,
    key: KeyValue,
) -> ExtendedOperation:
    """Return Identity if distance is 0, otherwise wrap operation in Inner."""
    if distance == 0:
        return Identity(key)
    return Inner(key, frozenset({operation}))
