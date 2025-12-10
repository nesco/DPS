"""
Sequence edit distance using dynamic programming.

Cost models:
- Edit: delete costs bit_length(element), add costs bit_length(element)
- MDL: delete costs log2(collection_size), add costs bit_length(element)
"""

import math
from functools import cache
from typing import Callable, Literal, Sequence

from localtypes import BitLengthAware, KeyValue

from .operations import (
    Add,
    Delete,
    ExtendedOperation,
    identity_or_inner,
)


@cache
def sequence_edit_distance(
    source_sequence: Sequence[BitLengthAware],
    target_sequence: Sequence[BitLengthAware],
    distance_func: Callable[
        [BitLengthAware, BitLengthAware],
        tuple[int, ExtendedOperation],
    ],
    len_func: Callable[[BitLengthAware], int],
    key: str | None = None,
    metric: Literal["edit", "mdl"] = "edit",
) -> tuple[int, Sequence[ExtendedOperation]]:
    """
    Computes the edit distance between two sequences using dynamic programming.

    Args:
        source_sequence: Source sequence.
        target_sequence: Target sequence.
        distance_func: Returns (distance, operation) between two elements.
        len_func: Cost of adding an element (typically bit_length).
        key: Field name containing this collection, if applicable.
        metric: "edit" for transformation distance, "mdl" for information distance.

    Returns:
        Tuple of (distance, list of operations).
    """
    m, n = len(source_sequence), len(target_sequence)

    # Delete cost depends on metric
    # MDL: cost to identify which element = log2(collection_size)
    # Edit: cost is the content being removed
    mdl_cost = math.ceil(math.log2(m)) if m > 1 else 0

    def get_delete_cost(elem: BitLengthAware) -> int:
        return mdl_cost if metric == "mdl" else len_func(elem)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    dist_cache: dict[tuple[int, int], tuple[int, ExtendedOperation]] = {}

    # Initialize first row and column with cumulative deletion/insertion costs
    for i in range(m + 1):
        dp[i][0] = sum(get_delete_cost(source_sequence[k]) for k in range(i))
    for j in range(n + 1):
        dp[0][j] = sum(len_func(target_sequence[k]) for k in range(j))

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            del_cost = dp[i - 1][j] + get_delete_cost(source_sequence[i - 1])
            ins_cost = dp[i][j - 1] + len_func(target_sequence[j - 1])
            sub_dist, sub_op = distance_func(
                source_sequence[i - 1], target_sequence[j - 1]
            )
            dist_cache[(i, j)] = (sub_dist, sub_op)
            sub_cost = dp[i - 1][j - 1] + sub_dist
            dp[i][j] = min(del_cost, ins_cost, sub_cost)

    # Backtrack to find operations
    operations: list[ExtendedOperation] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            dist, op = dist_cache[(i, j)]
            if dp[i][j] == dp[i - 1][j - 1] + dist:
                operations.append(identity_or_inner(dist, op, KeyValue((key, i - 1))))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + get_delete_cost(source_sequence[i - 1]):
            operations.append(Delete(KeyValue((key, i - 1)), source_sequence[i - 1]))
            i -= 1
        else:
            operations.append(Add(KeyValue((key, j - 1)), target_sequence[j - 1]))
            j -= 1

    operations.reverse()
    return dp[m][n], tuple(operations)
