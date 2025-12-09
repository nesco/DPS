"""
Pattern matching and unification for Kolmogorov Tree.

Functions:
    matches(pattern, subtree)      - Check if pattern matches, return bindings
    unify(pattern, subtree, bindings) - Recursive unification with binding updates
    abstract_node(index, pattern, node) - Replace matching node with SymbolNode
    node_to_symbolized_node(index, pattern, tree) - Symbolize all matches in tree

Types:
    Bindings = dict[int, BitLengthAware]  - Variable index -> bound value
"""

from typing import cast

from localtypes import BitLengthAware, Primitive

from kolmogorov_tree.nodes import (
    KNode,
    NestedNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    SymbolNode,
    VariableNode,
)
from kolmogorov_tree.primitives import IndexValue, T
from kolmogorov_tree.transformations import postmap

Bindings = dict[int, BitLengthAware]


def matches(pattern: KNode[T], subtree: KNode[T]) -> Bindings | None:
    """Returns variable bindings if pattern matches subtree, None otherwise."""
    if not isinstance(pattern, KNode) or not isinstance(subtree, KNode):
        raise TypeError("Both pattern and subtree must be KNode instances.")
    bindings: Bindings = {}
    if unify(pattern, subtree, bindings):
        return bindings
    return None


def _can_potentially_unify(pattern: KNode[T], subtree: KNode[T]) -> bool:
    """Quick type check for early pruning. VariableNodes can match anything."""
    if isinstance(pattern, VariableNode):
        return True
    return isinstance(subtree, type(pattern))


def unify_sum_children(
    p_idx: int,
    pattern_list: list[KNode[T]],
    subtree_list: list[KNode[T]],
    subtree_used: list[bool],
    bindings: Bindings,
    compatibility: list[list[bool]] | None = None,
) -> Bindings | None:
    """
    Backtracking search for bijective matching between SumNode children.

    Uses pre-computed compatibility matrix for early pruning.
    Returns bindings if complete matching found, None otherwise.
    """
    if p_idx == len(pattern_list):
        return bindings

    p_child = pattern_list[p_idx]

    for s_idx in range(len(subtree_list)):
        if subtree_used[s_idx]:
            continue

        if compatibility is not None and not compatibility[p_idx][s_idx]:
            continue

        s_child = subtree_list[s_idx]
        current_bindings = bindings.copy()

        if unify(p_child, s_child, current_bindings):
            subtree_used[s_idx] = True

            result = unify_sum_children(
                p_idx + 1,
                pattern_list,
                subtree_list,
                subtree_used,
                current_bindings,
                compatibility,
            )

            if result is not None:
                return result

            subtree_used[s_idx] = False

    return None


def unify(pattern: BitLengthAware, subtree: BitLengthAware, bindings: Bindings) -> bool:
    """
    Unifies pattern with subtree, updating bindings in place.

    Handles VariableNode binding, structural matching for all node types,
    and recursive descent through children.
    """
    if isinstance(pattern, VariableNode):
        index = pattern.index.value
        if index in bindings:
            return bindings[index] == subtree
        bindings[index] = subtree
        return True

    if not isinstance(subtree, type(pattern)):
        return False

    match pattern:
        case Primitive(value):
            assert isinstance(subtree, Primitive)
            return value == subtree.value

        case SumNode(children):
            assert isinstance(subtree, SumNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            if not children:
                return True

            sorted_pattern = sorted(children, key=str)
            sorted_subtree = sorted(s_children, key=str)
            subtree_used = [False] * len(sorted_subtree)

            compatibility = [
                [_can_potentially_unify(p, s) for s in sorted_subtree]
                for p in sorted_pattern
            ]

            initial_bindings = bindings.copy()
            result = unify_sum_children(
                0, sorted_pattern, sorted_subtree, subtree_used,
                initial_bindings, compatibility,
            )

            if result is not None:
                bindings.clear()
                bindings.update(result)
                return True
            return False

        case ProductNode(children):
            assert isinstance(subtree, ProductNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            for p_child, s_child in zip(children, s_children):
                if not unify(p_child, s_child, bindings):
                    return False
            return True

        case RepeatNode(node, count):
            subtree = cast(RepeatNode, subtree)
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != subtree.count:
                        return False
                else:
                    bindings[index] = subtree.count
            elif count != subtree.count:
                return False
            return unify(node, subtree.node, bindings)

        case NestedNode(index, node, count):
            subtree = cast(NestedNode, subtree)
            if isinstance(count, VariableNode):
                var_index = count.index.value
                if var_index in bindings:
                    if bindings[var_index] != subtree.count:
                        return False
                else:
                    bindings[var_index] = subtree.count
            elif count != subtree.count:
                return False
            return unify(node, subtree.node, bindings)

        case RootNode(node, position, colors):
            subtree = cast(RootNode, subtree)
            if not unify(position, subtree.position, bindings):
                return False
            if not unify(colors, subtree.colors, bindings):
                return False
            return unify(node, subtree.node, bindings)

        case SymbolNode(index, parameters):
            subtree = cast(SymbolNode, subtree)
            if index != subtree.index or len(parameters) != len(subtree.parameters):
                return False
            for param, s_param in zip(parameters, subtree.parameters):
                if not unify(param, s_param, bindings):
                    return False
            return True

        case RectNode(height, width):
            subtree = cast(RectNode, subtree)
            if not unify(height, subtree.height, bindings):
                return False
            return unify(width, subtree.width, bindings)

        case _:
            return pattern == subtree


def abstract_node(index: IndexValue, pattern: KNode[T], node: KNode[T]) -> KNode[T]:
    """Replaces node with SymbolNode if it matches pattern, else returns unchanged."""
    if pattern == node:
        return SymbolNode(index, ())

    bindings = matches(pattern, node)
    if bindings is not None:
        params = tuple(
            binding
            for j in range(max(bindings.keys(), default=-1) + 1)
            if (binding := bindings.get(j, None)) is not None
        )
        return SymbolNode(index, params)

    return node


def node_to_symbolized_node(
    index: IndexValue, pattern: KNode[T], knode: KNode[T]
) -> KNode[T]:
    """Replaces all occurrences of pattern in knode with SymbolNodes."""
    return postmap(knode, lambda node: abstract_node(index, pattern, node))


__all__ = [
    "Bindings",
    "matches",
    "unify",
    "unify_sum_children",
    "abstract_node",
    "node_to_symbolized_node",
]
