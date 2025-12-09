"""
Traversal utilities for the Kolmogorov Tree.

This module provides:
- get_subvalues: Get all BitLengthAware subvalues of a dataclass
- children: Get child KNodes of a node
- depth_first_preorder_bitlengthaware: DFS traversal for BitLengthAware
- breadth_first_preorder_knode: BFS traversal for KNodes
- next_layer: Get next layer of children (for BFS-like traversal)
- depth: Calculate tree depth
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from localtypes import BitLengthAware
from utils.tree_functionals import dataclass_subvalues, depth_first_preorder

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kolmogorov_tree.nodes import KNode
    from kolmogorov_tree.primitives import T


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


def children(knode: "KNode") -> "Iterator[KNode]":
    """Unified API to access children of standard KNodes nodes"""
    # Deferred import to avoid circular dependency
    from kolmogorov_tree.nodes import KNode

    subvalues = get_subvalues(knode)
    return (sv for sv in subvalues if isinstance(sv, KNode))


def depth_first_preorder_bitlengthaware(
    root: BitLengthAware,
) -> Iterator[BitLengthAware]:
    """Depth-first preorder traversal for BitLengthAware objects."""
    return depth_first_preorder(get_subvalues, root)


def breadth_first_preorder_knode(node: "KNode[T] | None") -> "Iterator[KNode[T]]":
    """Breadth-first preorder traversal for KNode objects."""
    return depth_first_preorder(children, node)


def next_layer(layer: "Iterable[KNode]") -> "tuple[KNode, ...]":
    """Used for BFS-like traversal of a K-Tree. It's basically `children` for iterable"""
    return tuple(child for node in layer for child in children(node))


def depth(node: "KNode") -> int:
    """Returns the depth of a Kolmogorov tree"""
    max_depth = 0
    layer: tuple[KNode, ...] = (node,)
    while layer:
        max_depth += 1
        layer = next_layer(layer)

    return max_depth


__all__ = [
    "get_subvalues",
    "children",
    "depth_first_preorder_bitlengthaware",
    "breadth_first_preorder_knode",
    "next_layer",
    "depth",
]
