"""
Set edit distance using the Hungarian algorithm.

Cost models:
- Edit: delete costs bit_length(element), add costs bit_length(element)
- MDL: delete costs log2(collection_size), add costs bit_length(element)
"""

import math
from collections.abc import Set
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from localtypes import BitLengthAware, KeyValue

from .operations import (
    Add,
    Delete,
    ExtendedOperation,
    identity_or_inner,
)


def set_edit_distance(
    source_set: Set[BitLengthAware],
    target_set: Set[BitLengthAware],
    distance_func: Callable[
        [BitLengthAware, BitLengthAware],
        tuple[int, ExtendedOperation],
    ],
    len_func: Callable[[BitLengthAware], int],
    key: str | None = None,
    metric: Literal["edit", "mdl"] = "edit",
) -> tuple[int, Sequence[ExtendedOperation]]:
    """
    Calculates the optimal set edit distance using the Hungarian algorithm.

    Args:
        source_set: Source set of elements.
        target_set: Target set of elements.
        distance_func: Returns (cost, operation) for substitution.
        len_func: Cost of adding an element (typically bit_length).
        key: Field name containing this collection, if applicable.
        metric: "edit" for transformation distance, "mdl" for information distance.

    Returns:
        Tuple of (distance, list of operations).
    """
    source_list = sorted(list(source_set), key=str)
    target_list = sorted(list(target_set), key=str)

    n = len(source_list)
    m = len(target_list)

    # Delete cost depends on metric
    if metric == "mdl":
        delete_cost = math.ceil(math.log2(n)) if n > 1 else 0
    else:
        delete_cost = None  # Will use len_func per element

    def get_delete_cost(elem: BitLengthAware) -> int:
        if delete_cost is not None:
            return delete_cost
        return len_func(elem)

    # Handle empty set cases
    key_val = KeyValue(key)
    if n == 0 and m == 0:
        return 0, tuple()
    if n == 0:
        total = sum(len_func(elem) for elem in target_list)
        ops = tuple(Add(key_val, elem) for elem in target_list)
        return total, ops
    if m == 0:
        total = sum(get_delete_cost(elem) for elem in source_list)
        ops = tuple(Delete(key_val, elem) for elem in source_list)
        return total, ops

    # Create cost matrix with dimensions (n+m) x (n+m)
    cost_matrix = np.full((n + m, n + m), np.inf)

    # Substitution costs (source_i -> target_j)
    for i in range(n):
        for j in range(m):
            cost, _ = distance_func(source_list[i], target_list[j])
            cost_matrix[i, j] = cost

    # Deletion costs (source_i -> dummy_target_i)
    for i in range(n):
        cost_matrix[i, m + i] = get_delete_cost(source_list[i])

    # Addition costs (dummy_source_j -> target_j)
    for j in range(m):
        cost_matrix[n + j, j] = len_func(target_list[j])

    # Dummy-to-dummy matches (zero cost, allows skips)
    for i in range(m):
        for j in range(n):
            cost_matrix[n + i, m + j] = 0.0

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute minimum cost
    valid_indices = np.where(cost_matrix[row_ind, col_ind] != np.inf)
    min_cost = cost_matrix[row_ind[valid_indices], col_ind[valid_indices]].sum()

    # Build operations from assignment
    operations: list[ExtendedOperation] = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            # Match (Substitution/Identity/Inner)
            dist, op = distance_func(source_list[r], target_list[c])
            operations.append(identity_or_inner(dist, op, key_val))
        elif r < n and c >= m:
            # Deletion
            operations.append(Delete(key_val, source_list[r]))
        elif r >= n and c < m:
            # Addition
            operations.append(Add(key_val, target_list[c]))
        # else: dummy match, ignore

    return int(np.round(min_cost)), tuple(operations)
