"""
Variable substitution and symbol resolution for Kolmogorov Tree.

Functions:
    substitute_variables(node, params) - Replace VariableNodes with concrete values
    reduce_abstraction(abstraction, params) - Apply parameters to an abstraction
    resolve_symbols(node, symbols)     - Expand SymbolNodes using symbol table

    Nesting:
        extract_nested_patterns(table, tree) - Find recursive patterns, create NestedNodes
        expand_nested_node(node, table)      - Expand single NestedNode
        expand_all_nested_nodes(node, table) - Expand all NestedNodes in tree
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
    Substitutes variables including those in non-traversed positions.

    .. deprecated:: Use substitute_variables instead.
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
    """Looks up variable value in params, returns node unchanged if index out of range."""
    if node.index.value < len(params):
        return params[node.index.value]
    return node


def substitute_variables(knode: KNode[T], params: Parameters) -> KNode[T]:
    """
    Recursively substitutes VariableNodes with values from params.

    Handles all field types: single values, tuples, and frozensets.
    """
    if isinstance(knode, VariableNode):
        node = variable_to_param(knode, params)
        if not isinstance(node, KNode):
            raise TypeError(f"{node} is not a KNode")
        return node

    nfields = {}
    for field_name, value in vars(knode).items():
        match value:
            case VariableNode():
                nvalue = variable_to_param(value, params)
            case frozenset():
                nvalue = frozenset(
                    substitute_variables(el, params) if isinstance(el, KNode) else el
                    for el in value
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
    """Applies parameters to an abstraction (cached for repeated calls)."""
    return substitute_variables(abstraction, params)


@functools.cache
def resolve_symbols(knode: KNode[T], symbols: Sequence[KNode[T]]) -> KNode[T]:
    """
    Expands all SymbolNodes using the symbol table.

    Handles nested symbol references by resolving until no SymbolNodes remain.
    """

    def resolve_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and 0 <= node.index.value < len(symbols):
            return reduce_abstraction(symbols[node.index.value], node.parameters)
        return node

    def resolve_until(node: KNode[T]) -> KNode[T]:
        while isinstance(node, SymbolNode) and node.index.value < len(symbols):
            node = resolve_f(node)
        return node

    return premap(knode, resolve_until, factorize=False)


def extract_nested_patterns(
    symbol_table: list[KNode[T]],
    tree: KNode[T],
) -> KNode[T]:
    """
    Finds recursive collection patterns and replaces them with NestedNodes.

    Mutates symbol_table by appending extracted templates.
    Only creates NestedNode if it reduces total bit length.
    """

    def mapping_function(node: KNode[T]) -> KNode[T]:
        if isinstance(node, (SumNode, ProductNode)):
            result = nested_collection_to_nested_node(node)
            if result is not None:
                nested_node, template = result
                if nested_node.bit_length() + template.bit_length() <= node.bit_length():
                    if template in symbol_table:
                        index = symbol_table.index(template)
                    else:
                        index = len(symbol_table)
                        symbol_table.append(template)
                    return NestedNode(
                        IndexValue(index), nested_node.node, nested_node.count
                    )
        return node

    return premap(tree, mapping_function)


def expand_nested_node(
    nested_node: NestedNode[T], symbol_table: Sequence[KNode[T]]
) -> KNode[T]:
    """Expands a single NestedNode by applying its template count times."""
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
    """Expands all NestedNodes in a tree using the symbol table."""

    def expand_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, NestedNode):
            return expand_nested_node(node, symbol_table)
        return node

    return postmap(knode, expand_f)
