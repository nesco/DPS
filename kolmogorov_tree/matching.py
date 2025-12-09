"""
Pattern matching and unification utilities for Kolmogorov Tree.

This module provides functions for pattern matching and unification between
KNodes, enabling symbolic abstraction through variable binding and substitution.
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
    """
    Determines if a pattern matches a subtree, returning variable bindings if successful.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree to match against.

    Returns:
        Bindings | None: A dictionary of variable bindings if the match succeeds, or None if it fails.
    """
    if not isinstance(pattern, KNode) or not isinstance(subtree, KNode):
        raise TypeError("Both pattern and subtree must be KNode instances.")
    bindings: Bindings = {}
    if unify(pattern, subtree, bindings):
        return bindings
    return None


def unify_sum_children(
    p_idx: int,  # Index of the current pattern child being matched
    pattern_list: list[KNode[T]],  # Sorted list of pattern children
    subtree_list: list[KNode[T]],  # Sorted list of subtree children
    subtree_used: list[bool],  # Tracks which subtree children are already matched
    bindings: Bindings,
) -> Bindings | None:
    """
    Helper function using backtracking on sorted lists to find a deterministic
    unification mapping between SumNode children.

    Args:
        p_idx: Current index in pattern_list being processed.
        pattern_list: Sorted list of pattern children.
        subtree_list: Sorted list of subtree children.
        subtree_used: Boolean mask indicating used subtree children.
        bindings: Current accumulated bindings for this search path.

    Returns:
        Updated bindings if a valid matching is found, otherwise None.
    """
    # Base case: All pattern children have been successfully matched.
    if p_idx == len(pattern_list):
        return bindings

    p_child = pattern_list[p_idx]

    # Iterate through potential matches in the sorted subtree list
    for s_idx in range(len(subtree_list)):
        # Check if this subtree child is already used in the current mapping
        if not subtree_used[s_idx]:
            s_child = subtree_list[s_idx]

            # Create a copy of bindings to explore this specific pair's unification
            current_bindings = bindings.copy()

            # Attempt to unify the chosen pair (p_child, s_child)
            if unify(p_child, s_child, current_bindings):
                # If the pair unifies successfully:
                # 1. Mark the subtree child as used for this path
                subtree_used[s_idx] = True

                # 2. Recursively try to match the *next* pattern child (p_idx + 1)
                result_bindings = unify_sum_children(
                    p_idx + 1,
                    pattern_list,
                    subtree_list,
                    subtree_used,
                    current_bindings,  # Pass the updated bindings
                )

                # 3. Check if the recursive call was successful
                if result_bindings is not None:
                    # Success! A complete valid matching was found. Return it.
                    return result_bindings

                # 4. Backtrack: If the recursive call failed, unmark the
                #    subtree child so it can be tried with other pattern children
                #    in alternative branches of the search.
                subtree_used[s_idx] = False
                # Continue the loop to try matching p_child with the next unused s_child

    # If the loop completes without returning, it means p_child could not be
    # successfully matched with any *available* s_child in a way that allowed the
    # rest of the pattern children to be matched. Backtrack by returning None.
    return None


def unify(pattern: BitLengthAware, subtree: BitLengthAware, bindings: Bindings) -> bool:
    """
    Recursively unifies a pattern with a subtree, updating bindings in place.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree or primitive to match against.
        bindings: Current bindings of variables to subtrees or primitives, modified in place.

    Returns:
        bool: True if unification succeeds, False if it fails.
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
            # Inside the unify function...
        case SumNode(children):  # noqa: F811
            assert isinstance(subtree, SumNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            if not children:
                return True

            # 1. Create sorted lists based on a canonical key
            # Sort using the  string representation
            sorted_pattern_children = sorted(children, key=str)
            sorted_subtree_children = sorted(s_children, key=str)
            # 2. Initialize the tracking list for used subtree children
            subtree_used = [False] * len(sorted_subtree_children)

            # 3. Call the deterministic backtracking helper
            initial_bindings = bindings.copy()
            successful_bindings = unify_sum_children(
                0,  # Start matching the first pattern child (index 0)
                sorted_pattern_children,
                sorted_subtree_children,
                subtree_used,
                initial_bindings,
            )

            if successful_bindings is not None:
                # Update original bindings *in place*
                bindings.clear()
                bindings.update(successful_bindings)
                return True
            else:
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
            s_node = subtree.node
            s_count = subtree.count
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != s_count:
                        return False
                else:
                    bindings[index] = s_count
            elif count != s_count:
                return False
            return unify(node, s_node, bindings)
        case NestedNode(index, node, count):
            subtree = cast(NestedNode, subtree)
            s_node = subtree.node
            s_count = subtree.count
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != s_count:
                        return False
                else:
                    bindings[index] = s_count
            elif count != s_count:
                return False
            return unify(node, s_node, bindings)

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
            if not unify(width, subtree.width, bindings):
                return False
            return True
        case _:
            return pattern == subtree


def abstract_node(index: IndexValue, pattern: KNode[T], node: KNode[T]) -> KNode[T]:
    """
    If node matches the pattern, it replaces it by a SymbolNode with the right bindings.
    Else, it returns the node as is.
    """
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
    """
    Transform a KNode by replacing all occurrences of the pattern with SymbolNodes.
    """
    return postmap(knode, lambda node: abstract_node(index, pattern, node))


__all__ = [
    "Bindings",
    "matches",
    "unify",
    "unify_sum_children",
    "abstract_node",
    "node_to_symbolized_node",
]
