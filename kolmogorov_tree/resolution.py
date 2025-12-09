"""
Resolution utilities for the Kolmogorov Tree.

This module provides functions to resolve NestedNode and SymbolNode to their
concrete KNode representations using a symbol table. This separation from
the node classes eliminates circular dependencies.

Functions:
- resolve: Resolve a NestedNode or SymbolNode to its concrete KNode
- eq_ref: Check if two resolvable nodes refer to the same symbol
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from localtypes import BitLengthAware

from kolmogorov_tree.nodes import NestedNode, SymbolNode

# Type variable for BitLengthAware, allowing the functions to work with both
# KNode and general BitLengthAware types used in edit.py
B = TypeVar("B", bound=BitLengthAware)


def resolve(node: B, symbol_table: Sequence[Any]) -> B:
    """
    Resolve a NestedNode or SymbolNode to its concrete KNode representation.

    For NestedNode: Expands the node using its template from the symbol table.
    For SymbolNode: Substitutes variable placeholders with the provided parameters.
    For other nodes: Returns the node unchanged.

    Args:
        node: The node to resolve (may be NestedNode, SymbolNode, or other).
        symbol_table: A sequence of templates indexed by symbol/nested index.

    Returns:
        The resolved node.
    """
    if isinstance(node, NestedNode):
        # Import here to avoid circular dependency
        from kolmogorov_tree.substitution import expand_nested_node

        return expand_nested_node(node, symbol_table)  # type: ignore[return-value]

    if isinstance(node, SymbolNode):
        # Import here to avoid circular dependency
        from kolmogorov_tree.substitution import reduce_abstraction

        return reduce_abstraction(symbol_table[node.index.value], node.parameters)  # type: ignore[return-value]

    return node


def eq_ref(a: Any, b: Any) -> bool:
    """
    Check if two nodes refer to the same symbol/nested index.

    This is used to detect when two SymbolNodes or NestedNodes point to the
    same template in the symbol table, allowing optimizations in edit distance
    calculations.

    Args:
        a: First node to compare.
        b: Second node to compare.

    Returns:
        True if both nodes are of the same resolvable type and have the same index.
    """
    if isinstance(a, NestedNode) and isinstance(b, NestedNode):
        return a.index == b.index
    if isinstance(a, SymbolNode) and isinstance(b, SymbolNode):
        return a.index == b.index
    return False


def is_resolvable(node: Any) -> bool:
    """
    Check if a node is resolvable (NestedNode or SymbolNode).

    Args:
        node: The node to check.

    Returns:
        True if the node is a NestedNode or SymbolNode.
    """
    return isinstance(node, (NestedNode, SymbolNode))


__all__ = [
    "resolve",
    "eq_ref",
    "is_resolvable",
]
