"""
Kolmogorov Tree: A bit-length-aware AST for representing non-deterministic programs.

This package implements a tree structure for representing non-deterministic programs,
particularly useful for describing shapes using grid movements (for ARC-AGI).
The representation aims to approximate Kolmogorov complexity through minimum description length.

The tree structure consists of:
- ProductNodes for deterministic sequences
- SumNodes for non-deterministic branching
- RepeatNodes for repetition extraction
- NestedNodes for nested patterns
- SymbolNodes for pattern abstraction and reuse
- Variable binding for lambda abstraction

Key Features:
- Computable bit-length metrics for complexity approximation
- Pattern extraction and memorization via symbol table
- Support for 8-directional grid movements
- Lambda abstraction through variable binding

Example Usage:
    >>> from kolmogorov_tree import create_moves_sequence, RepeatNode, SumNode, create_rect
    >>> pattern = create_moves_sequence("2323")  # Right-Down-Right-Down
    >>> repeated = RepeatNode(pattern, CountValue(3))
    >>> program = SumNode(frozenset([repeated, create_rect(3, 3)]))
"""

from __future__ import annotations

# Primitives - bit-length constants and value types
from kolmogorov_tree.primitives import (
    ARCBitLength,
    Alphabet,
    BitLength,
    CoordValue,
    CountValue,
    IndexValue,
    MoveValue,
    NoneValue,
    PaletteValue,
    T,
    VariableValue,
)

# Node classes - the core AST types
from kolmogorov_tree.nodes import (
    CollectionNode,
    KNode,
    NestedNode,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    SymbolNode,
    Uncompressed,
    Unsymbolized,
    VariableNode,
)

# Traversal utilities
from kolmogorov_tree.traversal import (
    breadth_first_preorder_knode,  # Deprecated alias
    children,
    depth,
    depth_first_preorder_bitlengthaware,
    get_subvalues,
    next_layer,
    preorder_knode,
)

# Factory functions for creating nodes
from kolmogorov_tree.factories import (
    create_move_node,
    create_moves_sequence,
    create_rect,
    create_variable_node,
    cv,
)

# Parsing utilities
from kolmogorov_tree.parsing import (
    split_top_level_arguments,
    str_to_knode,
    str_to_repr,
)

# Predicate/inspection utilities
from kolmogorov_tree.predicates import (
    arity,
    contained_symbols,
    is_abstraction,
    is_symbolized,
)

# Resolution utilities
from kolmogorov_tree.resolution import (
    eq_ref,
    is_resolvable,
    resolve,
)

# Transformation utilities
from kolmogorov_tree.transformations import (
    construct_product_node,
    encode_run_length,
    expand_repeats,
    factorize_tuple,
    find_repeating_pattern,
    flatten_product,
    flatten_sum,
    get_iterator,
    iterable_to_product,
    iterable_to_sum,
    postmap,
    premap,
    reconstruct_knode,
    reverse_node,
    shift,
    shift_f,
)

# Template extraction utilities
from kolmogorov_tree.templates import (
    Parameters,
    detect_recursive_collection,
    extract_nested_product_template,
    extract_nested_sum_template,
    extract_template,
    nested_collection_to_nested_node,
)

# Pattern matching/unification utilities
from kolmogorov_tree.matching import (
    Bindings,
    abstract_node,
    matches,
    node_to_symbolized_node,
    unify,
    unify_sum_children,
)

# Variable substitution utilities
from kolmogorov_tree.substitution import (
    expand_all_nested_nodes,
    expand_nested_node,
    extract_nested_patterns,
    reduce_abstraction,
    resolve_symbols,
    substitute_variables,
    substitute_variables_deprecated,
    variable_to_param,
)

# Symbolization utilities
from kolmogorov_tree.symbolization import (
    factor_by_existing_symbols,
    find_symbol_candidates,
    full_symbolization,
    greedy_symbolization,
    merge_symbol_tables,
    remap_sub_symbols,
    remap_symbol_indices,
    symbolize,
    symbolize_pattern,
    symbolize_together,
    unsymbolize,
    unsymbolize_all,
)

__all__ = [
    # Primitives
    "ARCBitLength",
    "Alphabet",
    "BitLength",
    "CoordValue",
    "CountValue",
    "IndexValue",
    "MoveValue",
    "NoneValue",
    "PaletteValue",
    "T",
    "VariableValue",
    # Nodes
    "CollectionNode",
    "KNode",
    "NestedNode",
    "PrimitiveNode",
    "ProductNode",
    "RectNode",
    "RepeatNode",
    "RootNode",
    "SumNode",
    "SymbolNode",
    "Uncompressed",
    "Unsymbolized",
    "VariableNode",
    # Traversal
    "breadth_first_preorder_knode",  # Deprecated alias
    "children",
    "depth",
    "depth_first_preorder_bitlengthaware",
    "get_subvalues",
    "next_layer",
    "preorder_knode",
    # Factories
    "create_move_node",
    "create_moves_sequence",
    "create_rect",
    "create_variable_node",
    "cv",
    # Parsing
    "split_top_level_arguments",
    "str_to_knode",
    "str_to_repr",
    # Predicates
    "arity",
    "contained_symbols",
    "is_abstraction",
    "is_symbolized",
    # Resolution
    "eq_ref",
    "is_resolvable",
    "resolve",
    # Transformations
    "construct_product_node",
    "encode_run_length",
    "expand_repeats",
    "factorize_tuple",
    "find_repeating_pattern",
    "flatten_product",
    "flatten_sum",
    "get_iterator",
    "iterable_to_product",
    "iterable_to_sum",
    "postmap",
    "premap",
    "reconstruct_knode",
    "reverse_node",
    "shift",
    "shift_f",
    # Templates
    "Parameters",
    "detect_recursive_collection",
    "extract_nested_product_template",
    "extract_nested_sum_template",
    "extract_template",
    "nested_collection_to_nested_node",
    # Matching
    "Bindings",
    "abstract_node",
    "matches",
    "node_to_symbolized_node",
    "unify",
    "unify_sum_children",
    # Substitution
    "expand_all_nested_nodes",
    "expand_nested_node",
    "extract_nested_patterns",
    "reduce_abstraction",
    "resolve_symbols",
    "substitute_variables",
    "substitute_variables_deprecated",
    "variable_to_param",
    # Symbolization
    "factor_by_existing_symbols",
    "find_symbol_candidates",
    "full_symbolization",
    "greedy_symbolization",
    "merge_symbol_tables",
    "remap_sub_symbols",
    "remap_symbol_indices",
    "symbolize",
    "symbolize_pattern",
    "symbolize_together",
    "unsymbolize",
    "unsymbolize_all",
]
