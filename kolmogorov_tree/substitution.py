"""
Variable substitution and symbol resolution utilities for Kolmogorov Tree.

This module provides functions for substituting variables in KNodes,
reducing abstractions, and resolving symbolic references.
"""

import functools
from typing import Sequence

from localtypes import BitLengthAware

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
from kolmogorov_tree.primitives import (
    CoordValue,
    CountValue,
    IndexValue,
    PaletteValue,
    T,
)
from kolmogorov_tree.templates import (
    Parameters,
    nested_collection_to_nested_node,
)
from kolmogorov_tree.transformations import postmap, premap


def substitute_variables_deprecated(
    abstraction: KNode[T], params: Parameters
) -> KNode[T]:
    """
    Helper function to substitute variables, including those that are not traversed
    during a simple node traversal.

    .. deprecated::
        Use :func:`substitute_variables` instead.
    """
    match abstraction:
        case VariableNode(index) if index.value < len(params):
            node = params[index.value]
            if not isinstance(node, KNode):
                raise TypeError(
                    f"Trying to substitute a non-node parameter to a variable encountered during node traversal: {[node]}"
                )
            return node
        case RepeatNode(node, VariableNode(index)) if index.value < len(params):
            count = params[index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return RepeatNode(node, count)
        case NestedNode(index, node, count) if isinstance(
            count, VariableNode
        ) and count.index.value < len(params):
            count = params[count.index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return NestedNode(index, node, count)
        case RootNode(node, VariableNode(index1), VariableNode(index2)) if (
            index1.value < len(params) and index2.value < len(params)
        ):
            position = params[index1.value]
            colors = params[index2.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, VariableNode(index), colors) if index.value < len(params):
            position = params[index.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, position, VariableNode(index)) if index.value < len(params):
            colors = params[index.value]
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RectNode(VariableNode(index1), VariableNode(index2)) if index1.value < len(
            params
        ) and index2.value < len(params):
            height = params[index1.value]
            width = params[index2.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case RectNode(VariableNode(index), width) if index.value < len(params):
            height = params[index.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            return RectNode(height, width)
        case RectNode(height, VariableNode(index)) if index.value < len(params):
            width = params[index.value]
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case _:
            return abstraction


def variable_to_param(node: VariableNode, params: Parameters) -> BitLengthAware:
    if node.index.value < len(params):
        value = params[node.index.value]
        return value
    return node
    # raise ValueError(f"Unknown variable {node.index.value} for {params}")


def substitute_variables(knode: KNode[T], params: Parameters) -> KNode[T]:
    # Avoid infinite recursion on VariableNodes by treating them immediately
    # And returning directly
    if isinstance(knode, VariableNode):
        node = variable_to_param(knode, params)
        if not isinstance(node, KNode):
            raise TypeError(f"{node} is not a KNode")
        return node

    # Iterating over fields
    nfields = {}
    for field_name, value in vars(knode).items():
        match value:
            # Replace directly variables to avoid hitting the TypeError
            case VariableNode():
                nvalue = variable_to_param(value, params)
            case frozenset():
                nvalue = frozenset(
                    {
                        substitute_variables(el, params)
                        if isinstance(el, KNode)
                        else el
                        for el in value
                    }
                )
            case tuple():
                nvalue = tuple(
                    substitute_variables(el, params) if isinstance(el, KNode) else el
                    for el in value
                )
            case KNode():
                nvalue = substitute_variables(value, params)
            case _:
                nvalue = value

        nfields[field_name] = nvalue

    return type(knode)(**nfields)


@functools.cache
def reduce_abstraction(abstraction: KNode[T], params: Parameters) -> KNode[T]:
    """
    Substitutes variable placeholders in the template with the corresponding parameters.

    This function recursively traverses the template node and replaces VariableNodes
    with the provided parameters. It also handles composite nodes by substituting variables
    in their children or attributes.

    Args:
        abstraction (KNode): The abstract node to process.
        params (tuple[Any, ...]): The parameters to substitute for variables.

    Returns:
        KNode: The abstraction with variables replaced by parameters.
    """
    # return postmap(abstraction, lambda node: substitute_variables(node, params))
    return substitute_variables(abstraction, params)


@functools.cache
def resolve_symbols(knode: KNode[T], symbols: Sequence[KNode[T]]) -> KNode[T]:
    """
    Resolves symbolic references recursively using the symbol table.

    This function traverses the Kolmogorov Tree and replaces SymbolNodes with their
    definitions from the symbol table, handling parameter substitution. It recursively
    resolves all sub-nodes for composite node types.

    Args:
        node (KNode): The node to resolve.
        symbols (list[KNode]): The list of symbol definitions.

    Returns:
        KNode: The resolved node with symbols expanded.
    """

    def resolve_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and 0 <= node.index.value < len(symbols):
            return reduce_abstraction(symbols[node.index.value], node.parameters)
        return node

    # A symbolic node can be resolved into another symbolic node
    def resolve_until(node: KNode[T]) -> KNode[T]:
        while isinstance(node, SymbolNode) and node.index.value < len(symbols):
            node = resolve_f(node)
        return node

    node = premap(knode, resolve_until, factorize=False)

    return node


def extract_nested_patterns(
    symbol_table: list[KNode[T]],
    tree: KNode[T],
) -> KNode[T]:
    """
    Traverses a Kolmogorov Tree to identify and replace nested patterns with NestedNodes,
    extracting their templates into a symbol table.

    Args:
        tree: The input Kolmogorov Tree (KNode[T]).

    Returns:
        KNode[T]: The transformed tree with NestedNodes replacing nested patterns.

    Edge-Effects:
        Fill the given symbol table with templates extracted from the nested patterns.
    """

    def mapping_function(node: KNode[T]) -> KNode[T]:
        """
        Mapping function for postmap to detect nested patterns and replace them with NestedNodes.

        Args:
            node: The current node being processed.

        Returns:
            KNode[T]: The transformed node (either a NestedNode or the original node).
        """
        # Only process SumNode or ProductNode for nested patterns
        if isinstance(node, (SumNode, ProductNode)):
            result = nested_collection_to_nested_node(node)
            if result is not None:
                nested_node, template = result
                # Greedy test, it might be better because the template might be shared with others
                if (
                    nested_node.bit_length() + template.bit_length()
                    <= node.bit_length()
                ):
                    # Add the template to the symbol table and get its index
                    if template in symbol_table:
                        index = symbol_table.index(template)
                    else:
                        index = len(symbol_table)
                        symbol_table.append(template)
                    # Create a new NestedNode with the correct index
                    # Since NestedNode is frozen, instantiate a new one
                    new_nested_node = NestedNode(
                        IndexValue(index), nested_node.node, nested_node.count
                    )
                    return new_nested_node
        # Return the node unchanged if no pattern is detected
        return node

    # Apply postmap to transform the tree
    new_tree = premap(tree, mapping_function)
    return new_tree


def expand_nested_node(
    nested_node: NestedNode[T], symbol_table: Sequence[KNode[T]]
) -> KNode[T]:
    """Expands a NestedNode using its template from the symbol table."""
    template = symbol_table[nested_node.index.value]
    current = nested_node.node

    if not isinstance(nested_node.count, CountValue):
        raise ValueError(
            f"Trying to expand a nested node with a variable count: {nested_node}"
        )

    for _ in range(nested_node.count.value):
        current = postmap(template, lambda node: substitute_variables(node, (current,)))
    return current


def expand_all_nested_nodes(knode: KNode[T], symbol_table: Sequence[KNode[T]]):
    """
    Expand all the inner NestedNodes of a given node.

    Args:
        knode: KNode possibly containing nested nodes
        symbol_table: Symbol Table containing all the template referenced by NestedNode. It will fail otherwise.
    Returns:
        KNode[T]: node free of NestedNodes
    """

    def expand_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, NestedNode):
            return expand_nested_node(node, symbol_table)
        return node

    return postmap(knode, expand_f)
