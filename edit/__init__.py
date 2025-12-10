"""
Edit distance and transformation module.

This module computes edit distance and transformations between tree-like data structures.
Four valid operations are used: Identity, Add, Delete, Substitute.
Extended operations add: Prune, Graft, Inner, Resolve.

Distances are computed with given distance and "length" functions.
"""

from .operations import (
    Add,
    Delete,
    ExtendedOperation,
    Graft,
    Identity,
    Inner,
    Operation,
    Prune,
    Resolve,
    Substitute,
    identity_or_inner,
)
from .sequence import sequence_edit_distance
from .sets import set_edit_distance
from .mdl import (
    # MDL distance (zero-cost deletes)
    mdl_distance,
    mdl_distance_value,
    symmetric_mdl_distance,
    # Edit distance (thermodynamic deletes)
    edit_distance,
    edit_distance_value,
    symmetric_edit_distance,
    # Joint/normalized metrics
    joint_complexity,
    normalized_compression_distance,
    normalized_information_distance,
    # Aliases for backward compatibility
    conditional_complexity,
    min_distance,
    symmetric_distance,
)
from .transform import TransformationError, apply_transformation
from .tree import (
    build_nested_graft,
    build_nested_prune,
    collect_all_descendants,
    collect_links,
    compute_bit_length,
    extended_edit_distance,
    recursive_edit_distance,
)
from .normalized import (
    # Normalization functions
    normalize_position,
    normalize_color,
    normalize_counts,
    extract_structure,
    extract_template,
    # Normalized distance functions
    structural_distance,
    structural_distance_value,
    template_distance,
    template_distance_value,
    scale_normalized_distance,
    abstract_match_score,
)

__all__ = [
    # Operations
    "Operation",
    "ExtendedOperation",
    "Identity",
    "Add",
    "Delete",
    "Substitute",
    "Inner",
    "Prune",
    "Graft",
    "Resolve",
    "identity_or_inner",
    # Distance functions
    "sequence_edit_distance",
    "set_edit_distance",
    "recursive_edit_distance",
    "extended_edit_distance",
    # MDL distance (zero-cost deletes)
    "mdl_distance",
    "mdl_distance_value",
    "symmetric_mdl_distance",
    # Edit distance (thermodynamic deletes)
    "edit_distance",
    "edit_distance_value",
    "symmetric_edit_distance",
    # Joint/normalized metrics
    "joint_complexity",
    "normalized_information_distance",
    "normalized_compression_distance",
    # Aliases
    "symmetric_distance",
    "min_distance",
    "conditional_complexity",
    # Transform
    "apply_transformation",
    "TransformationError",
    # Utilities
    "compute_bit_length",
    "collect_links",
    "collect_all_descendants",
    "build_nested_prune",
    "build_nested_graft",
    # Normalized distance (abstract matching)
    "normalize_position",
    "normalize_color",
    "normalize_counts",
    "extract_structure",
    "extract_template",
    "structural_distance",
    "structural_distance_value",
    "template_distance",
    "template_distance_value",
    "scale_normalized_distance",
    "abstract_match_score",
]
