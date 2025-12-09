"""
Kolmogorov Tree: A bit-length-aware AST for representing non-deterministic programs.

This package implements a tree structure for representing non-deterministic programs,
particularly useful for describing shapes using grid movements (for ARC-AGI).
The representation aims to approximate Kolmogorov complexity through minimum description length.

For now, this re-exports everything from the original monolithic module.
"""

# Re-export primitives
from kolmogorov_tree.primitives import (  # noqa: F401
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

# Re-export nodes
from kolmogorov_tree.nodes import (  # noqa: F401
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

# Re-export traversal utilities
from kolmogorov_tree.traversal import (  # noqa: F401
    breadth_first_preorder_knode,
    children,
    depth,
    depth_first_preorder_bitlengthaware,
    get_subvalues,
    next_layer,
)

# Re-export factory functions
from kolmogorov_tree.factories import (  # noqa: F401
    create_move_node,
    create_moves_sequence,
    create_rect,
    create_variable_node,
    cv,
)

# Re-export parsing utilities
from kolmogorov_tree.parsing import (  # noqa: F401
    split_top_level_arguments,
    str_to_knode,
    str_to_repr,
)

# Re-export predicate/inspection utilities
from kolmogorov_tree.predicates import (  # noqa: F401
    arity,
    contained_symbols,
    is_abstraction,
    is_symbolized,
)

# Re-export everything else from the legacy module for backwards compatibility
from kolmogorov_tree_legacy import *  # noqa: F401, F403
