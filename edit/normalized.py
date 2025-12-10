"""
Normalized distance for abstract structural matching.

The standard edit distance is too literal - it penalizes:
- Different positions (coordinates should be relative, not absolute)
- Different colors (color is often arbitrary in ARC)
- Different scales (same pattern at different sizes should be similar)

This module provides normalized distance that factors out these differences.
"""

from dataclasses import fields, is_dataclass, replace
from typing import Sequence

from kolmogorov_tree import (
    CoordValue,
    CountValue,
    KNode,
    MoveValue,
    NoneValue,
    PaletteValue,
    PrimitiveNode,
    ProductNode,
    RepeatNode,
    RootNode,
    SumNode,
    VariableNode,
)
from arc.types import Coord
from kolmogorov_tree.types import BitLengthAware, Primitive

from .tree import extended_edit_distance, recursive_edit_distance


def normalize_position(node: KNode) -> KNode:
    """
    Normalize positions by translating to origin.

    Replaces CoordValue with a canonical (0, 0) position.
    This makes distance independent of absolute position.
    """
    if isinstance(node, RootNode):
        return RootNode(
            node.node,
            CoordValue(Coord(0, 0)),  # Normalize to origin
            node.colors,
        )
    return node


def normalize_color(node: KNode, color_map: dict[int, int] | None = None) -> KNode:
    """
    Normalize colors by mapping to canonical identifiers.

    All colors become 0, 1, 2, ... in order of first appearance.
    This makes distance independent of actual color values.
    """
    if color_map is None:
        color_map = {}

    def get_canonical_color(c: int) -> int:
        if c not in color_map:
            color_map[c] = len(color_map)
        return color_map[c]

    if isinstance(node, RootNode):
        if isinstance(node.colors, PaletteValue):
            new_colors = frozenset(get_canonical_color(c) for c in node.colors.value)
            inner = node.node
            if inner is not None and not isinstance(inner, NoneValue):
                inner = normalize_color(inner, color_map)
            return RootNode(
                inner,
                node.position,
                PaletteValue(new_colors),
            )
    elif isinstance(node, ProductNode):
        return ProductNode(
            tuple(normalize_color(child, color_map) for child in node.children)
        )
    elif isinstance(node, SumNode):
        return SumNode(
            frozenset(normalize_color(child, color_map) for child in node.children)
        )

    return node


def extract_structure(node: KNode) -> KNode:
    """
    Extract pure structure by removing position and color information.

    This gives the "shape signature" of the node - what pattern it represents
    independent of where it is or what color it is.
    """
    node = normalize_position(node)
    node = normalize_color(node)
    return node


def structural_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> tuple[int, object]:
    """
    Compute distance based on structure only.

    Normalizes both source and target before comparison:
    - Positions translated to origin
    - Colors mapped to canonical identifiers

    This makes markers with different positions/colors have distance 0.
    """
    source_norm = extract_structure(source) if isinstance(source, KNode) else source
    target_norm = extract_structure(target) if isinstance(target, KNode) else target

    if use_extended:
        return extended_edit_distance(source_norm, target_norm, symbol_table)
    else:
        return recursive_edit_distance(source_norm, target_norm, symbol_table)


def structural_distance_value(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Structural distance returning only the numeric value."""
    return float(structural_distance(source, target, symbol_table, use_extended)[0])


def scale_normalized_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
) -> float:
    """
    Distance that accounts for scale differences.

    Two objects with the same pattern but different sizes (e.g., (0)*{3} vs (0)*{5})
    should have lower distance than objects with completely different patterns.

    This uses the ratio of structural distance to total complexity.
    """
    struct_dist = structural_distance_value(source, target, symbol_table)

    # Normalize by the larger complexity
    max_complexity = max(source.bit_length(), target.bit_length())
    if max_complexity == 0:
        return 0.0

    return struct_dist / max_complexity


def normalize_counts(node: KNode) -> KNode:
    """
    Normalize repeat counts to a canonical value.

    Replaces all CountValue with a canonical count (e.g., 1).
    This makes patterns like (0)*{3} and (0)*{5} identical.
    """
    if isinstance(node, RepeatNode):
        normalized_inner = normalize_counts(node.node)
        # Use canonical count of 1
        return RepeatNode(normalized_inner, CountValue(1))
    elif isinstance(node, ProductNode):
        return ProductNode(tuple(normalize_counts(child) for child in node.children))
    elif isinstance(node, SumNode):
        return SumNode(frozenset(normalize_counts(child) for child in node.children))
    elif isinstance(node, RootNode):
        inner = node.node
        if inner is not None and not isinstance(inner, NoneValue):
            inner = normalize_counts(inner)
        return RootNode(inner, node.position, node.colors)
    return node


def extract_template(node: KNode) -> KNode:
    """
    Extract the abstract template of a node.

    Normalizes position, color, AND repeat counts.
    This gives the pure "pattern shape" regardless of size.

    Example: Both (0)*{3} and (0)*{5} become (0)*{1}
    """
    node = normalize_position(node)
    node = normalize_color(node)
    node = normalize_counts(node)
    return node


def template_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> tuple[int, object]:
    """
    Compute distance based on template only.

    Normalizes position, color, AND repeat counts before comparison.
    This makes patterns with the same shape but different sizes have distance 0.
    """
    source_norm = extract_template(source) if isinstance(source, KNode) else source
    target_norm = extract_template(target) if isinstance(target, KNode) else target

    if use_extended:
        return extended_edit_distance(source_norm, target_norm, symbol_table)
    else:
        return recursive_edit_distance(source_norm, target_norm, symbol_table)


def template_distance_value(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Template distance returning only the numeric value."""
    return float(template_distance(source, target, symbol_table, use_extended)[0])


def abstract_match_score(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    position_weight: float = 0.0,  # Ignore position differences by default
    color_weight: float = 0.0,  # Ignore color differences by default
    scale_weight: float = 0.5,  # Partial weight for scale differences
) -> float:
    """
    Compute abstract matching score with configurable weights.

    Lower score = better match.

    Args:
        source, target: Objects to compare
        symbol_table: For resolving symbol references
        position_weight: How much position differences matter (0 = ignore)
        color_weight: How much color differences matter (0 = ignore)
        scale_weight: How much scale differences matter (0 = ignore, 1 = full)

    Returns:
        Matching score (lower = more similar)
    """
    # Full structural distance (no normalization)
    full_dist, _ = extended_edit_distance(source, target, symbol_table)

    # Structural distance (normalized)
    struct_dist = structural_distance_value(source, target, symbol_table)

    # The difference is what position/color/scale contribute
    normalization_cost = full_dist - struct_dist

    # Weighted combination
    # struct_dist captures pattern differences
    # normalization_cost captures position/color/scale differences
    weighted_dist = (
        struct_dist
        + (position_weight + color_weight + scale_weight) / 3 * normalization_cost
    )

    return weighted_dist
