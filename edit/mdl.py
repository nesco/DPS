"""
Edit distance and MDL distance metrics.

Two distance metrics:

1. **Edit Distance**: Transformation distance
   - Delete costs bit_length(element)
   - Prune costs bit_length difference
   - Good for measuring transformation magnitude

2. **MDL Distance**: Information distance (Minimum Description Length)
   - Delete costs log2(collection_size) to identify which element
   - Prune costs log2(path_space) to identify which path
   - Good for measuring K(B|A) - bits to describe B given A
"""

from typing import Literal, Sequence

from kolmogorov_tree.types import BitLengthAware

from .tree import extended_edit_distance, recursive_edit_distance


def edit_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> tuple[int, object]:
    """
    Edit distance: transformation cost between source and target.

    Delete costs bit_length (content being removed).
    Prune costs bit_length difference (structure being removed).

    Args:
        source: Source object.
        target: Target object.
        symbol_table: Symbols for resolving references.
        use_extended: If True, use Prune/Graft operations.

    Returns:
        Tuple of (distance, operations).
    """
    if use_extended:
        return extended_edit_distance(source, target, symbol_table, "edit")
    else:
        return recursive_edit_distance(source, target, symbol_table, True, "edit")


def edit_distance_value(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Edit distance returning only the numeric value."""
    return float(edit_distance(source, target, symbol_table, use_extended)[0])


def mdl_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> tuple[int, object]:
    """
    MDL distance: K(target | source) - bits to describe target given source.

    Delete costs log2(collection_size) to identify which element.
    Prune costs log2(path_space) to identify which descendant path.

    Args:
        source: Source object (what receiver knows).
        target: Target object (what to describe).
        symbol_table: Symbols for resolving references.
        use_extended: If True, use Prune/Graft operations.

    Returns:
        Tuple of (distance, operations).
    """
    if use_extended:
        return extended_edit_distance(source, target, symbol_table, "mdl")
    else:
        return recursive_edit_distance(source, target, symbol_table, True, "mdl")


def mdl_distance_value(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """MDL distance returning only the numeric value."""
    return float(mdl_distance(source, target, symbol_table, use_extended)[0])


def symmetric_edit_distance(
    a: BitLengthAware,
    b: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Symmetric edit distance: d(A→B) + d(B→A)."""
    d_ab = edit_distance_value(a, b, symbol_table, use_extended)
    d_ba = edit_distance_value(b, a, symbol_table, use_extended)
    return d_ab + d_ba


def symmetric_mdl_distance(
    a: BitLengthAware,
    b: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Symmetric MDL distance: d(A→B) + d(B→A)."""
    d_ab = mdl_distance_value(a, b, symbol_table, use_extended)
    d_ba = mdl_distance_value(b, a, symbol_table, use_extended)
    return d_ab + d_ba


def joint_complexity(
    elements: Sequence[BitLengthAware],
    symbol_table: Sequence[BitLengthAware] = (),
    metric: Literal["edit", "mdl"] = "edit",
    use_extended: bool = True,
) -> float:
    """
    Joint complexity K(A, B, C, ...).

    Computed as: K(A) + K(B|A) + K(C|A,B) + ...
    Approximates K(X|A,B,...) as min(K(X|A), K(X|B), ...).

    Use case: Scoring cliques - prefer groups with shared structure.

    Args:
        elements: Sequence of BitLengthAware objects.
        symbol_table: Symbols for resolving references.
        metric: "edit" or "mdl".
        use_extended: If True, use Prune/Graft operations.

    Returns:
        Joint complexity as float.
    """
    if not elements:
        return 0.0

    dist_func = mdl_distance_value if metric == "mdl" else edit_distance_value

    # Sort by complexity (smallest first)
    sorted_elems = sorted(elements, key=lambda x: x.bit_length())

    # K(first element)
    total = float(sorted_elems[0].bit_length())

    # K(elem_i | elem_0, ..., elem_{i-1}) ≈ min distance to any previous
    for i in range(1, len(sorted_elems)):
        current = sorted_elems[i]
        min_cond = float("inf")
        for j in range(i):
            prev = sorted_elems[j]
            cond = dist_func(prev, current, symbol_table, use_extended)
            min_cond = min(min_cond, cond)
        total += min_cond

    return total


def normalized_information_distance(
    a: BitLengthAware,
    b: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    metric: Literal["edit", "mdl"] = "edit",
    use_extended: bool = True,
) -> float:
    """
    Normalized Information Distance (NID).

    NID = (K(A|B) + K(B|A)) / K(A,B)

    Range approximately [0, 1]. NID ≈ 0 means similar, NID ≈ 1 means different.
    """
    dist_func = mdl_distance_value if metric == "mdl" else edit_distance_value

    d_ab = dist_func(a, b, symbol_table, use_extended)
    d_ba = dist_func(b, a, symbol_table, use_extended)

    k_a = a.bit_length()
    k_ab = k_a + d_ab

    if k_ab == 0:
        return 0.0

    return (d_ab + d_ba) / k_ab


def normalized_compression_distance(
    a: BitLengthAware,
    b: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    metric: Literal["edit", "mdl"] = "edit",
    use_extended: bool = True,
) -> float:
    """
    Normalized Compression Distance (NCD).

    NCD = (K(A,B) - min(K(A), K(B))) / max(K(A), K(B))

    Range approximately [0, 1+ε]. NCD ≈ 0 means containment, NCD ≈ 1 means different.
    """
    dist_func = mdl_distance_value if metric == "mdl" else edit_distance_value

    k_a = a.bit_length()
    k_b = b.bit_length()
    d_ab = dist_func(a, b, symbol_table, use_extended)

    k_ab = k_a + d_ab
    min_k = min(k_a, k_b)
    max_k = max(k_a, k_b)

    if max_k == 0:
        return 0.0

    return (k_ab - min_k) / max_k


# Backward-compatible aliases
symmetric_distance = symmetric_edit_distance
conditional_complexity = mdl_distance_value


def min_distance(
    a: BitLengthAware,
    b: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    use_extended: bool = True,
) -> float:
    """Minimum of forward and backward edit distance."""
    return min(
        edit_distance_value(a, b, symbol_table, use_extended),
        edit_distance_value(b, a, symbol_table, use_extended),
    )
