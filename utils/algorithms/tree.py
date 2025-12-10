"""
Tree traversal and transformation utilities.

Traversals:
    breadth_first_preorder(after, root)  - BFS yielding nodes level by level
    breadth_first_postorder(after, root) - BFS yielding deepest levels first
    depth_first_preorder(after, root)    - DFS yielding parent before children
    depth_first_postorder(after, root)   - DFS yielding children before parent

Mappings:
    postorder_map(node, f, after, reconstruct) - Transform tree bottom-up
    preorder_map(node, f, after, reconstruct)  - Transform tree top-down

Dataclass Utilities:
    dataclass_subvalues(obj) - Yield all field values (flattening collections)
    breadth_first_preorder_bitlengthaware(root) - BFS for BitLengthAware trees

Hashing:
    cached_hash(obj)         - Hash with id-based caching
    build_hash_hierarchy(obj) - Map each subtree hash to its children's hashes

Note: For graphs with cycles, add `seen = set()` to avoid infinite loops.
"""

from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Callable, Generic, Iterator, Sequence, TypeVar

from collections.abc import Hashable

if TYPE_CHECKING:
    from kolmogorov_tree.types import BitLengthAware

T = TypeVar("T")


# =============================================================================
# Test Data Structure
# =============================================================================


@dataclass(frozen=True)
class RoseNode(Generic[T]):
    """Generic tree node for testing. Each node has a value and children."""

    value: T
    children: "tuple[RoseNode[T], ...]" = field(default_factory=tuple)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__


# =============================================================================
# Traversals
# =============================================================================


def breadth_first_preorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """Yields nodes level by level, root first."""
    if root is None:
        return
    queue = deque([root])
    while queue:
        current = queue.popleft()
        yield current
        for child in after(current):
            queue.append(child)


def breadth_first_postorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """Yields nodes level by level, deepest level first."""
    if root is None:
        return

    levels: dict[int, list[T]] = {}
    queue: deque[tuple[T, int]] = deque([(root, 0)])
    seen: set[int] = {id(root)}

    while queue:
        node, level = queue.popleft()
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

        for child in after(node):
            child_id = id(child)
            if child_id not in seen:
                seen.add(child_id)
                queue.append((child, level + 1))

    for level in sorted(levels.keys(), reverse=True):
        yield from levels[level]


def depth_first_preorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """Yields parent before children, depth-first."""
    if root is None:
        return
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        children = list(after(current))
        stack.extend(reversed(children))


def depth_first_postorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """Yields children before parent, depth-first."""
    if root is None:
        return
    stack: list[tuple[T, bool]] = [(root, False)]
    while stack:
        current, visited = stack.pop()
        if visited:
            yield current
        else:
            stack.append((current, True))
            children = list(after(current))
            for child in reversed(children):
                stack.append((child, False))


# =============================================================================
# Mappings
# =============================================================================


def postorder_map(
    node: T,
    f: Callable[[T], T],
    after: Callable[[T], Iterator[T]],
    reconstruct: Callable[[T, Sequence[T]], T],
) -> T:
    """
    Transforms tree bottom-up: children processed before parent.

    Args:
        node: Root of the tree
        f: Transform function applied after children are processed
        after: Returns iterator of node's children
        reconstruct: Rebuilds node with new children
    """
    new_children = tuple(
        postorder_map(child, f, after, reconstruct) for child in after(node)
    )
    new_node = reconstruct(node, new_children)
    return f(new_node)


def preorder_map(
    node: T,
    f: Callable[[T], T],
    after: Callable[[T], Iterator[T]],
    reconstruct: Callable[[T, Sequence[T]], T],
) -> T:
    """
    Transforms tree top-down: parent processed before children.

    Args:
        node: Root of the tree
        f: Transform function applied before children are processed
        after: Returns iterator of node's children
        reconstruct: Rebuilds node with new children
    """
    transformed = f(node)
    new_children = [
        preorder_map(child, f, after, reconstruct) for child in after(transformed)
    ]
    return reconstruct(transformed, new_children)


# =============================================================================
# Dataclass Utilities
# =============================================================================


def dataclass_subvalues(obj: object) -> "Iterator[BitLengthAware]":
    """
    Yields all field values from a dataclass, flattening collections.

    For tuple/list/set/frozenset fields, yields each element individually.
    """
    if not is_dataclass(obj):
        return
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, (tuple, list, set, frozenset)):
            yield from value
        else:
            yield value


def breadth_first_preorder_bitlengthaware(
    root: "BitLengthAware",
) -> "Iterator[BitLengthAware]":
    """BFS traversal for BitLengthAware dataclass trees."""
    return breadth_first_preorder(dataclass_subvalues, root)


# =============================================================================
# Hashing
# =============================================================================

_hash_cache: dict[int, int] = {}


def cached_hash(obj: Hashable) -> int:
    """
    Returns hash with id-based caching.

    Useful when hashing the same objects repeatedly during their lifetime.
    """
    obj_id = id(obj)
    if obj_id not in _hash_cache:
        _hash_cache[obj_id] = hash(obj)
    return _hash_cache[obj_id]


def build_hash_hierarchy(obj: object) -> dict[int, frozenset[int]]:
    """
    Maps each subtree's hash to the hashes of its immediate children.

    Traverses a tree of dataclasses and collections, building a hash hierarchy
    useful for structural comparisons.
    """
    hash_map: dict[int, frozenset[int]] = {}

    def traverse(node: object) -> None:
        if not is_dataclass(node):
            if isinstance(node, (tuple, frozenset)):
                for elem in node:
                    traverse(elem)
            else:
                hash_map[cached_hash(node)] = frozenset()
        else:
            child_hashes = [
                cached_hash(getattr(node, field.name)) for field in fields(node)
            ]
            hash_map[cached_hash(node)] = frozenset(child_hashes)
            for field in fields(node):
                traverse(getattr(node, field.name))

    traverse(obj)
    return hash_map


__all__ = [
    "RoseNode",
    "breadth_first_preorder",
    "breadth_first_postorder",
    "depth_first_preorder",
    "depth_first_postorder",
    "postorder_map",
    "preorder_map",
    "dataclass_subvalues",
    "breadth_first_preorder_bitlengthaware",
    "cached_hash",
    "build_hash_hierarchy",
]
