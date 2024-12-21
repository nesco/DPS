"""
Implements rose trees
"""

from typing import Generic, TypeVar, Callable
from dataclasses import dataclass, field

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class RoseTree(Generic[T]):
    value: T
    children: tuple["RoseTree[T]", ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        value_len = len(self.value) if hasattr(self.value, "__len__") else 1
        return value_len + sum(len(child) for child in self.children)

    def __iter__(self):
        """Iterate over all values in a depth-first manner."""
        yield self.value
        for child in self.children:
            yield from child


class Rose:
    @staticmethod
    def is_leaf(rose_tree: RoseTree) -> bool:
        return len(rose_tree) == 0

    @staticmethod
    def leaves(rose_tree: RoseTree[T]) -> list[T]:
        """Return a list of all leaf values (nodes without children)."""

        if not rose_tree.children:
            return [rose_tree.value]
        return [leaf for child in rose_tree.children for leaf in Rose.leaves(child)]

    @staticmethod
    def count(rose_tree: RoseTree) -> int:
        return 1 + sum(Rose.count(child) for child in rose_tree.children)

    @staticmethod
    def height(rose_tree: RoseTree) -> int:
        return 1 + max(Rose.height(child) for child in rose_tree.children)

    @staticmethod
    def map(rose_tree: RoseTree[T], f: Callable[[T], U]) -> RoseTree[U]:
        """Map a function over every value in a rose tree."""

        return RoseTree(
            f(rose_tree.value),
            tuple(Rose.map(child, f) for child in rose_tree.children),
        )

    @staticmethod
    def fold(rose_tree: RoseTree[T], f: Callable[[T, list[U]], U]) -> U:
        """
        Fold over the tree, applying f to each node's value and its folded children.
        The function f takes a value and list of child results and returns a result.
        """
        child_results = [Rose.fold(child, f) for child in rose_tree.children]
        return f(rose_tree.value, child_results)

    @staticmethod
    def breadth_first(rose_tree):
        """Iterate over all values in a breadth-first manner."""
        from collections import deque

        queue = deque([rose_tree])
        while queue:
            node = queue.popleft()
            yield node.value
            queue.extend(node.children)

    @staticmethod
    def find(rose_tree: RoseTree[T], predicate: Callable[[T], bool]) -> T | None:
        """
        Find first value in the tree that matches the predicate (depth-first search).
        Returns None if no match is found.
        """
        if predicate(rose_tree.value):
            return rose_tree.value

        for child in rose_tree.children:
            result = Rose.find(child, predicate)
            if result is not None:
                return result

        return None

    @staticmethod
    def to_dict(rose_tree: RoseTree[T]) -> dict:
        """Convert a rose tree to a dictionary representation."""
        return {
            "value": rose_tree.value,
            "children": [Rose.to_dict(child) for child in rose_tree.children],
        }

    @staticmethod
    def from_dict(data: dict) -> RoseTree:
        """Create a rose tree from a dictionary representation."""
        return RoseTree(
            value=data["value"],
            children=tuple(Rose.from_dict(child) for child in data["children"]),
        )
