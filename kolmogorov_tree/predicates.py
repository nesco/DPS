"""
Predicate functions for Kolmogorov Tree inspection.

Functions:
    is_symbolized(node)      - True if node contains SymbolNodes
    is_abstraction(node)     - True if node contains VariableNodes
    contained_symbols(node)  - All symbol indices in subtree
    arity(node)              - Max variable index + 1 (abstraction arity)
"""

from __future__ import annotations

from functools import cache

from kolmogorov_tree.nodes import KNode, SymbolNode, VariableNode
from kolmogorov_tree.primitives import IndexValue, VariableValue
from kolmogorov_tree.traversal import (
    breadth_first_preorder_knode,
    depth_first_preorder_bitlengthaware,
    get_subvalues,
)


@cache
def is_symbolized(node: KNode) -> bool:
    """True if node or any descendant is a SymbolNode."""
    subnodes = breadth_first_preorder_knode(node)
    return any(isinstance(n, SymbolNode) for n in subnodes)


@cache
def is_abstraction(node: KNode) -> bool:
    """True if node or any descendant is a VariableNode/VariableValue."""
    sub_values = depth_first_preorder_bitlengthaware(node)
    return any(
        isinstance(value, VariableNode) or isinstance(value, VariableValue)
        for value in sub_values
    )


def contained_symbols(knode: KNode) -> tuple[IndexValue, ...]:
    """Returns all symbol table indices referenced in the subtree."""
    subnodes = breadth_first_preorder_knode(knode)
    return tuple(node.index for node in subnodes if isinstance(node, SymbolNode))


def arity(node: KNode) -> int:
    """Returns max variable index + 1 (number of parameters for an abstraction)."""
    subvalues = depth_first_preorder_bitlengthaware(node)
    variable_indices = [
        value.value for value in subvalues if isinstance(value, VariableValue)
    ]
    if variable_indices:
        return max(variable_indices) + 1
    return 0


__all__ = [
    "is_symbolized",
    "is_abstraction",
    "contained_symbols",
    "arity",
]
