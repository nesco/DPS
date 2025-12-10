"""
Tree traversal utilities for KNode structures.

Functions:
    children(node)      - Direct KNode children of a node
    preorder_knode(node) - DFS preorder iterator over all descendant KNodes
    next_layer(nodes)   - All children of a node collection (for BFS)
    depth(node)         - Maximum depth of a tree
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Iterator

from kolmogorov_tree.types import BitLengthAware
from utils.tree_functionals import dataclass_subvalues, depth_first_preorder

if TYPE_CHECKING:
    from kolmogorov_tree.nodes import KNode
    from kolmogorov_tree.primitives import T


def get_subvalues(obj: BitLengthAware) -> Iterator[BitLengthAware]:
    """Yields all BitLengthAware field values from a dataclass instance."""
    return dataclass_subvalues(obj)


def children(knode: "KNode") -> "Iterator[KNode]":
    """Yields direct KNode children of a node."""
    from kolmogorov_tree.nodes import KNode

    return (sv for sv in get_subvalues(knode) if isinstance(sv, KNode))


def depth_first_preorder_bitlengthaware(
    root: BitLengthAware,
) -> Iterator[BitLengthAware]:
    """DFS preorder traversal over all BitLengthAware descendants."""
    return depth_first_preorder(get_subvalues, root)


def preorder_knode(node: "KNode[T] | None") -> "Iterator[KNode[T]]":
    """DFS preorder traversal over all KNode descendants."""
    return depth_first_preorder(children, node)


# Deprecated alias
breadth_first_preorder_knode = preorder_knode


def next_layer(layer: "Iterable[KNode]") -> "tuple[KNode, ...]":
    """Returns all children of nodes in the given layer (for BFS traversal)."""
    return tuple(child for node in layer for child in children(node))


def depth(node: "KNode") -> int:
    """Returns the maximum depth of a tree (root = depth 1)."""
    current_depth = 0
    layer: tuple[KNode, ...] = (node,)
    while layer:
        current_depth += 1
        layer = next_layer(layer)
    return current_depth


__all__ = [
    "get_subvalues",
    "children",
    "depth_first_preorder_bitlengthaware",
    "preorder_knode",
    "breadth_first_preorder_knode",
    "next_layer",
    "depth",
]
