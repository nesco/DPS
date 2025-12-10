"""
ARC grid encoding/decoding using Kolmogorov syntax trees.

This module provides lossless encoding of ARC grids into abstract syntax trees (ASTs)
that represent shapes using Freeman chain codes. The encoding supports pattern
compression via RepeatNode and SymbolNode, enabling approximate Kolmogorov
complexity computation for representation selection.

Encoding Pipeline:
    Coords → FreemanNode → KNode → RootNode

Decoding Pipeline:
    RootNode → KNode → Coords → Points → Grid

Example:
    >>> from arc import encode_component, decode_root
    >>> from localtypes import Coord
    >>>
    >>> # Encode a rectangular component
    >>> component = {Coord(0, 0), Coord(1, 0), Coord(0, 1), Coord(1, 1)}
    >>> colors = {1}
    >>> distribution, symbols = encode_component(component, colors)
    >>>
    >>> # Decode back to points
    >>> root = distribution[0]  # Most compact encoding
    >>> points = decode_root(root)

Key Types:
    - RootNode: Top-level AST with position and color metadata
    - KNode: Abstract syntax tree nodes (ProductNode, SumNode, etc.)
    - MoveValue: Freeman chain code direction primitive
"""

# Encoding API
from .encoding import (
    # Primary encoding functions
    syntax_tree_at,
    encode_component_distribution,
    encode_component,
    # Lower-level encoding
    freeman_to_knode,
    encode_freeman_to_knode,
    create_root_node,
    # Utilities
    find_candidate_start_points,
    detect_rectangle_pattern,
    apply_rectangle_detection,
    # Composite structure
    UnionNode,
)

# Decoding API
from .decoding import (
    # Primary decoding functions
    decode_root,
    decode_knode,
    # Lower-level decoding
    execute_moves,
)

# Backward-compatible aliases (from original arc_syntax_tree.py)
from .encoding import (
    moves_to_rect,
    detect_rect_node,
    freeman_node_to_root_node,
    get_potential_starting_points,
    component_to_raw_syntax_tree_distribution,
    component_to_distribution,
)
from .decoding import decode_knode_coords

__all__ = [
    # Primary encoding
    "syntax_tree_at",
    "encode_component_distribution",
    "encode_component",
    # Primary decoding
    "decode_root",
    "decode_knode",
    "execute_moves",
    # Lower-level encoding
    "freeman_to_knode",
    "encode_freeman_to_knode",
    "create_root_node",
    "find_candidate_start_points",
    "detect_rectangle_pattern",
    "apply_rectangle_detection",
    # Composite
    "UnionNode",
    # Backward-compatible aliases
    "moves_to_rect",
    "detect_rect_node",
    "freeman_node_to_root_node",
    "get_potential_starting_points",
    "component_to_raw_syntax_tree_distribution",
    "component_to_distribution",
    "decode_knode_coords",
]
