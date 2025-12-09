"""
Predicate and inspection utilities for the Kolmogorov Tree.

This module provides functions to inspect and query KNodes:
- is_symbolized: Check if a node contains any SymbolNodes
- is_abstraction: Check if a node contains any VariableNodes
- contained_symbols: Get all symbol indices in a node
- arity: Get the arity (max variable index + 1) of an abstraction
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kolmogorov_tree.nodes import SymbolNode, VariableNode
from kolmogorov_tree.primitives import IndexValue, VariableValue
from kolmogorov_tree.traversal import (
    breadth_first_preorder_knode,
    depth_first_preorder_bitlengthaware,
    get_subvalues,
)

if TYPE_CHECKING:
    from kolmogorov_tree.nodes import KNode


def is_symbolized(node: "KNode") -> bool:
    """Return True if and only if node contains at least one SymbolNode in its subnodes."""
    subnodes = breadth_first_preorder_knode(node)
    return any(isinstance(n, SymbolNode) for n in subnodes)


def is_abstraction(node: "KNode") -> bool:
    """Return True if and only if node contains at least one VariableNode in its subvalues."""
    sub_values = get_subvalues(node)
    return any(
        isinstance(value, VariableNode) or isinstance(value, VariableValue)
        for value in sub_values
    )


def contained_symbols(knode: "KNode") -> tuple[IndexValue, ...]:
    """Get all symbol indices contained in a KNode."""
    subnodes = breadth_first_preorder_knode(knode)
    return tuple(node.index for node in subnodes if isinstance(node, SymbolNode))


def arity(node: "KNode") -> int:
    """Return the max index of the node variable, which is the arity of an abstraction."""
    subvalues = depth_first_preorder_bitlengthaware(node)
    variable_numbers = [
        value.value for value in subvalues if isinstance(value, VariableValue)
    ]
    if variable_numbers:
        return max(variable_numbers) + 1
    return 0


__all__ = [
    "is_symbolized",
    "is_abstraction",
    "contained_symbols",
    "arity",
]
