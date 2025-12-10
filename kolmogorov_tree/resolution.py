"""
Resolution utilities for Kolmogorov Tree.

Functions:
    resolve(node, table)   - Expand NestedNode/SymbolNode using symbol table
    eq_ref(a, b)           - Check if two nodes reference same symbol
    is_resolvable(node)    - Check if node is NestedNode or SymbolNode

This module exists to break circular dependencies between node classes
and substitution logic.
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from kolmogorov_tree.types import BitLengthAware

from kolmogorov_tree.nodes import NestedNode, SymbolNode

B = TypeVar("B", bound=BitLengthAware)


def resolve(node: B, symbol_table: Sequence[Any]) -> B:
    """
    Expands a NestedNode or SymbolNode to its concrete representation.

    Other node types are returned unchanged.
    """
    if isinstance(node, NestedNode):
        from kolmogorov_tree.substitution import expand_nested_node

        return expand_nested_node(node, symbol_table)  # type: ignore[return-value]

    if isinstance(node, SymbolNode):
        from kolmogorov_tree.substitution import reduce_abstraction

        return reduce_abstraction(symbol_table[node.index.value], node.parameters)  # type: ignore[return-value]

    return node


def eq_ref(a: Any, b: Any) -> bool:
    """True if both nodes are the same resolvable type with the same index."""
    if isinstance(a, NestedNode) and isinstance(b, NestedNode):
        return a.index == b.index
    if isinstance(a, SymbolNode) and isinstance(b, SymbolNode):
        return a.index == b.index
    return False


def is_resolvable(node: Any) -> bool:
    """True if node is a NestedNode or SymbolNode."""
    return isinstance(node, (NestedNode, SymbolNode))


__all__ = [
    "resolve",
    "eq_ref",
    "is_resolvable",
]
