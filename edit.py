"""
This module is dedicated to compute edit distance and transformations.
Four valid operations will be used: Identity, Add, Delete, Substitute.
Distances are computed with given distance and "length" functions.
If there is some kind of neutral element in the "set" of objects to compare, the "length" function will usually be the distance to it.

For tree type structures, Identity only mean the current non-tree values of the compared node are similar, as operations on their children will be linearized and thus counted

Note: For a bitlengthaware dataclass, an object is considered a leaf if and only if it inherits from Primitive
"""

from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cache
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment

from kolmogorov_tree.resolution import eq_ref, is_resolvable, resolve
from localtypes import (
    BitLengthAware,
    KeyValue,
    Primitive,
    ensure_all_instances,
)
from utils.tree_functionals import (
    RoseNode,
    breadth_first_preorder_bitlengthaware,
    # cached_hash,
)

cached_hash = hash

T = TypeVar("T")

type Operation = Add | Delete | Identity | Substitute | Inner

type ExtendedOperation = (
    Identity | Add | Delete | Substitute | Prune | Graft | Inner | Resolve
)


@dataclass(frozen=True)
class Inner(BitLengthAware):
    key: KeyValue
    children: "frozenset[ExtendedOperation]"

    def bit_length(self) -> int:
        return sum(op.bit_length() for op in self.children)

    def __str__(self) -> str:
        return f"Inner({self.children})"


@dataclass(frozen=True)
class Identity(BitLengthAware):
    """Represents an element that remains unchanged."""

    key: KeyValue
    # value: BitLengthAware

    def bit_length(self) -> int:
        return 0

    def __str__(self):
        return "Identity"  # . Original element: {self.value}"


@dataclass(frozen=True)
class Add(BitLengthAware):
    """Represents an element that is added to a collection."""

    key: KeyValue
    value: BitLengthAware

    def bit_length(self) -> int:
        # if isinstance(self.value, BitLengthAware):
        #     return self.value.bit_length()
        # return 1
        return self.value.bit_length()

    def __str__(self):
        return f"Add: {self.value} to a collection"


@dataclass(frozen=True)
class Delete(BitLengthAware):
    """Represents an element that is deleted in a collection."""

    key: KeyValue
    value: BitLengthAware  # Necessary for frozensets

    def bit_length(self) -> int:
        # if isinstance(self.value, BitLengthAware):
        #    return self.value.bit_length()
        # return 1
        return self.value.bit_length()

    def __str__(self):
        return f"Delete {self.value} {(f' ON key: {self.key}') if self.key is not None else ''}"


@dataclass(frozen=True)
class Substitute(BitLengthAware):
    """Represents an element substituted with another."""

    key: KeyValue
    # before: BitLengthAware
    after: BitLengthAware

    def bit_length(self) -> int:
        # if isinstance(self.before, BitLengthAware) and isinstance(
        #     self.after, BitLengthAware
        # ):
        #     return self.before.bit_length() + self.after.bit_length()
        return self.after.bit_length()

        # return 2

    def __str__(self):
        # return f"Substitute: {self.before} |-> {self.after} on key: {self.key}"
        return f"Substitute: object |-> {self.after}" + (
            f" On key: {self.key}" if self.key is not None else ""
        )


# Helpers
def identity_or_inner(
    source: BitLengthAware,
    target: BitLengthAware,
    sub_dist: int,
    sub_ops: ExtendedOperation,
    key: KeyValue,
) -> ExtendedOperation:
    """
    The aim of tree transformations is to use as much of inner transformations on subnodes instead of a general substitution as possible.
    """
    if sub_dist == 0:
        return Identity(key)

    return Inner(key, frozenset({sub_ops}))


@cache
def sequence_edit_distance(
    source_sequence: Sequence[BitLengthAware],
    target_sequence: Sequence[BitLengthAware],
    distance_func: Callable[
        [BitLengthAware, BitLengthAware],
        tuple[int, Operation],
    ],
    len_func: Callable[[BitLengthAware], int],
    key: str | None = None,
) -> tuple[int, Sequence[Operation]]:
    """
    Computes the edit distance between two sequences using dynamic programming.

    Args:
        source_sequence: First sequence.
        target_sequence: Second sequence.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.
        key: Name of the field containing the collection, if applicable

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    m, n = len(source_sequence), len(target_sequence)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column with cumulative deletion/insertion costs
    for i in range(m + 1):
        dp[i][0] = sum(len_func(source_sequence[k]) for k in range(i))
    for j in range(n + 1):
        dp[0][j] = sum(len_func(target_sequence[k]) for k in range(j))

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            delete_cost = dp[i - 1][j] + len_func(source_sequence[i - 1])
            insert_cost = dp[i][j - 1] + len_func(target_sequence[j - 1])
            sub_dist, _ = distance_func(
                source_sequence[i - 1], target_sequence[j - 1]
            )
            substitute_cost = dp[i - 1][j - 1] + sub_dist
            dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

    # Backtrack to find transformations if requested
    transformations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_dist, sub_ops = distance_func(
                source_sequence[i - 1], target_sequence[j - 1]
            )
            if dp[i][j] == dp[i - 1][j - 1] + sub_dist:
                operation = identity_or_inner(
                    source_sequence[i - 1],
                    target_sequence[j - 1],
                    sub_dist,
                    sub_ops,
                    KeyValue((key, i - 1)),
                )
                transformations.append(operation)
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + len_func(
            source_sequence[i - 1]
        ):
            transformations.append(
                # Delete(KeyValue(i - 1), source_sequence[i - 1])
                Delete(KeyValue((key, i - 1)), source_sequence[i - 1])
            )
            i -= 1
        else:
            # transformations.append(Add(KeyValue(j - 1), target_sequence[j - 1]))
            # transformations.append(Add(KeyValue(None), target_sequence[j - 1]))
            transformations.append(
                Add(KeyValue((key, j - 1)), target_sequence[j - 1])
            )
            j -= 1
    transformations.reverse()  # Reverse to get transformations in forward order

    return dp[m][n], tuple(transformations)


def set_edit_distance(
    source_set: Set[BitLengthAware],
    target_set: Set[BitLengthAware],
    distance_func: Callable[
        [BitLengthAware, BitLengthAware],
        tuple[int, Operation],
    ],
    len_func: Callable[[BitLengthAware], int],
    key: str | None = None,
) -> tuple[int, Sequence[Operation]]:
    """
    Calculates the optimal set edit distance using the Hungarian algorithm
    and returns the sequence of operations.

    Args:
        source_set: Set of elements in the source set.
        target_set: Set of elements in the target set.
        distance_func: Function(elem1, elem2) -> tuple[cost, operation]
                       calculating cost and base operation for substitution.
        len_func: Function(elem) -> cost of adding or deleting elem.
        key: Optional key associated with the set operation context.

    Returns:
        Tuple containing:
            - The minimum total edit distance (int).
            - A sequence of Operation objects detailing the optimal transformation.
    """
    # Convert sets to lists sorted by string representation for consistent ordering
    # Note: Sorting might impact performance for very large sets but ensures deterministic matrix construction
    source_list = sorted(list(source_set), key=str)
    target_list = sorted(list(target_set), key=str)

    n = len(source_list)
    m = len(target_list)

    # Handle empty set cases directly
    key_val = KeyValue(key)  # Create KeyValue once
    if n == 0 and m == 0:
        return 0, tuple()
    if n == 0:
        dist = sum(len_func(elem) for elem in target_list)
        ops = tuple(Add(key_val, elem) for elem in target_list)  #
        return dist, ops
    if m == 0:
        dist = sum(len_func(elem) for elem in source_list)
        ops = tuple(Delete(key_val, elem) for elem in source_list)  #
        return dist, ops

    # Create the cost matrix with dimensions (n+m) x (n+m)
    cost_matrix = np.full((n + m, n + m), np.inf)

    # 1. Substitution costs (source_i -> target_j)
    #    Use only the cost component for the assignment algorithm
    for i in range(n):
        for j in range(m):
            cost, _ = distance_func(source_list[i], target_list[j])
            cost_matrix[i, j] = cost

    # 2. Deletion costs (source_i -> dummy_target_i)
    for i in range(n):
        cost_matrix[i, m + i] = len_func(source_list[i])

    # 3. Addition costs (dummy_source_j -> target_j)
    for j in range(m):
        cost_matrix[n + j, j] = len_func(target_list[j])

    # 4. Cost of matching dummy source to dummy target (allows skips)
    for i in range(m):
        for j in range(n):
            cost_matrix[n + i, m + j] = 0.0

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # The minimum cost is the sum of the costs of the optimal assignment
    # Need to handle potential infinities if the assignment includes them (shouldn't happen with proper padding)
    valid_assignment_indices = np.where(cost_matrix[row_ind, col_ind] != np.inf)
    min_cost = cost_matrix[
        row_ind[valid_assignment_indices], col_ind[valid_assignment_indices]
    ].sum()

    # Determine operations based on the optimal assignment
    operations = []
    key_val = KeyValue(key)  #

    for r, c in zip(row_ind, col_ind):
        # Case 1: Match (Substitution/Identity/Inner)
        if r < n and c < m:
            source_elem = source_list[r]
            target_elem = target_list[c]
            # Recalculate distance_func to get the operation details
            sub_dist, sub_op = distance_func(source_elem, target_elem)
            # Use identity_or_inner helper to create the correct operation type
            final_op = identity_or_inner(
                source_elem, target_elem, sub_dist, sub_op, key_val
            )
            operations.append(final_op)

        # Case 2: Deletion
        elif r < n and c >= m:
            source_elem = source_list[r]
            operations.append(Delete(key_val, source_elem))  #

        # Case 3: Addition
        elif r >= n and c < m:
            target_elem = target_list[c]
            operations.append(Add(key_val, target_elem))  #

        # Case 4: Dummy match (r >= n and c >= m) - Ignore these

    # Return the total minimum cost and the sequence of operations
    return int(np.round(min_cost)), tuple(operations)


def set_edit_distance_deprecated(
    source_set: set[BitLengthAware] | frozenset[BitLengthAware],
    target_set: set[BitLengthAware] | frozenset[BitLengthAware],
    distance_func: Callable[
        [BitLengthAware, BitLengthAware],
        tuple[int, Operation],
    ],
    len_func: Callable[[BitLengthAware], int],
    key: str | None = None,
) -> tuple[int, Sequence[Operation]]:
    """
    Computes the edit distance between two sets using a greedy approach.

    Args:
        source_set: First set.
        target_set: Second set.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.
        path: path of the source set
        key: Name of the field containing the collection, if applicable

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    # Handle empty set cases
    if not source_set and not target_set:
        return 0, []
    if not source_set:
        dist = sum(len_func(elem) for elem in target_set)
        return dist, [Add(KeyValue(key), elem) for elem in target_set]
    if not target_set:
        dist = sum(len_func(elem) for elem in source_set)
        return dist, [Delete(KeyValue(key), elem) for elem in source_set]

    # Convert sets to lists for processing
    source_sequence = list(source_set)
    target_sequence = list(target_set)
    available = set(range(len(target_sequence)))
    total_dist = 0
    transformations = []
    key_val = KeyValue(key)

    # Greedy matching
    for source_elem in source_sequence:
        # Don't really matther for sets and stochastic so to remove
        if available:
            min_dist, min_j = min(
                (
                    distance_func(source_elem, target_sequence[j])[0],
                    j,
                )
                for j in available
            )
            if min_dist < len_func(source_elem) + len_func(
                target_sequence[min_j]
            ):
                total_dist += min_dist
                sub_dist, sub_ops = distance_func(
                    source_elem, target_sequence[min_j]
                )
                operation = identity_or_inner(
                    source_elem,
                    target_sequence[min_j],
                    sub_dist,
                    sub_ops,
                    key_val,
                )
                transformations.append(operation)
                available.remove(min_j)
            else:
                total_dist += len_func(source_elem)
                transformations.append(Delete(key_val, source_elem))
        else:
            total_dist += len_func(source_elem)
            transformations.append(Delete(key_val, source_elem))

    # Handle remaining elements in target_set
    for j in available:
        total_dist += len_func(target_sequence[j])
        transformations.append(Add(key_val, target_sequence[j]))

    return total_dist, tuple(transformations)


@dataclass(frozen=True)
class Prune(BitLengthAware):
    """When only the children element at key value is kept."""

    key: KeyValue
    then: ExtendedOperation
    # parent: BitLengthAware
    # child: BitLengthAware

    def bit_length(self) -> int:
        # return self.parent.bit_length() - self.child.bit_length()
        return self.key.bit_length() + self.then.bit_length()

    def __str__(self):
        if isinstance(self.key.value, tuple):
            field, index = self.key.value
            return f"Keeping element at key {field}[{index}] \n next: {self.then}"
        return f"Keeping element at key {self.key}\n next: {self.then}"


@dataclass(frozen=True)
class Graft(BitLengthAware):
    """When a children is added to a parent, it overwrites a parent *existing field*."""

    key: KeyValue
    parent: BitLengthAware
    first: ExtendedOperation
    # child: BitLengthAware

    def bit_length(self) -> int:
        # return self.parent.bit_length() - self.child.bit_length()
        return (
            self.key.bit_length()
            + self.parent.bit_length()
            + self.first.bit_length()
        )

    def __str__(self):
        return f"Attaching element at: {self.parent}[{self.key if self.key else 'None'}]\n Element: {self.first}"


@dataclass(frozen=True)
class Resolve(BitLengthAware):
    """
    SymbolNode → concrete subtree.  Carries the inner edit script
    computed on the resolved children so that callers can inspect it.
    """

    key: KeyValue
    inner: "ExtendedOperation"  # May be Identity/Inner/…

    def bit_length(self) -> int:
        return self.inner.bit_length()

    def __str__(self):
        return f"Resolving to {self.inner}"


def collect_links(
    bla: BitLengthAware,
    hash_to_object: dict[int, BitLengthAware],
    parent_to_children: defaultdict[int, set[int]],
    parent_field: dict[int, KeyValue],
):
    # Preorder collection of the relevant hashes
    bla_hash = cached_hash(bla)
    hash_to_object[bla_hash] = bla

    # Depth first propagation of the collection
    if is_dataclass(bla) and not isinstance(bla, Primitive):
        for field in fields(bla):
            attr = getattr(bla, field.name)
            if isinstance(attr, BitLengthAware):
                field_hash = cached_hash(attr)
                parent_to_children[bla_hash].add(field_hash)
                parent_field[field_hash] = KeyValue(field.name)
                collect_links(
                    attr, hash_to_object, parent_to_children, parent_field
                )
            if isinstance(attr, tuple):
                attr = ensure_all_instances(attr, BitLengthAware)
                for i, elem in enumerate(attr):
                    field_hash = cached_hash(elem)
                    parent_to_children[bla_hash].add(field_hash)
                    parent_field[field_hash] = KeyValue((field.name, i))
                    collect_links(
                        elem,
                        hash_to_object,
                        parent_to_children,
                        parent_field,
                    )
            if isinstance(attr, frozenset):
                attr = ensure_all_instances(attr, BitLengthAware)
                for elem in attr:
                    field_hash = cached_hash(elem)
                    parent_to_children[bla_hash].add(field_hash)
                    parent_field[field_hash] = KeyValue(field.name)
                    collect_links(
                        elem,
                        hash_to_object,
                        parent_to_children,
                        parent_field,
                    )


def compute_bit_length(obj: Any) -> int:
    if isinstance(obj, BitLengthAware):
        return obj.bit_length()
    elif isinstance(obj, (set, frozenset, tuple, list)):
        return sum(compute_bit_length(elem) for elem in obj)
    else:
        # Default for non-BitLengthAware primitives; adjust as needed
        return 1


# TO-DO: Find a way to allow several Prune operations without complexity explosion?
def extended_edit_distance(
    source: BitLengthAware | None,
    target: BitLengthAware | None,
    symbol_table: Sequence[BitLengthAware],
) -> tuple[int, ExtendedOperation | None]:
    """
    Computes the extended edit distance between two BitLengthAware tree-like objects.

    This function extends traditional edit distance by including Prune and Graft operations,
    making it suitable for determining if the target tree is a subtree of the source tree.
    It calculates the minimum cost to transform the source into the target using operations:
    Identity (no change), Add (insert), Delete (remove), Substitute (replace), Prune (keep only
    one child, removing others and parent value), and Graft (attach the current node to a root with other childs).

    Args:
        source: The source tree, a BitLengthAware object or None.
        target: The target tree, a BitLengthAware object or None.

    Returns:
        A tuple of (distance, operations), where distance is the minimal edit cost (in bits),
        and operations is a tuple of ExtendedOperation instances detailing the transformation.

    """

    # Handle resolvable nodes (SymbolNode, NestedNode)
    source_resolvable = source is not None and is_resolvable(source)
    target_resolvable = target is not None and is_resolvable(target)

    if source_resolvable and target_resolvable:
        assert source is not None and target is not None  # For type checker
        if eq_ref(source, target):
            pass  # Same reference, continue with normal comparison
        else:
            d, op = extended_edit_distance(
                resolve(source, symbol_table),
                resolve(target, symbol_table),
                symbol_table,
            )
            return d, Resolve(KeyValue(None), op) if op is not None else (d, None)  # type: ignore[return-value]
    elif source_resolvable:
        assert source is not None  # For type checker
        return extended_edit_distance(
            resolve(source, symbol_table),
            target,
            symbol_table,
        )
    elif target_resolvable:
        assert target is not None  # For type checker
        return extended_edit_distance(
            source,
            resolve(target, symbol_table),
            symbol_table,
        )

    if source is None and target is None:
        return 0, None  # Maybe Identity(tuple())?

    if source is None:
        assert target is not None  # Pyright
        return (target.bit_length(), Add(KeyValue(None), target))

    if target is None:
        return (source.bit_length(), Delete(KeyValue(None), source))

    # First collect the object topological structure and build the hash -> object map
    hash_to_object: dict[int, BitLengthAware] = {}
    parent_to_children_source: defaultdict[int, set[int]] = defaultdict(set)
    parent_to_children_target: defaultdict[int, set[int]] = defaultdict(set)
    parent_field: dict[int, KeyValue] = {}

    collect_links(
        source, hash_to_object, parent_to_children_source, parent_field
    )
    collect_links(
        target, hash_to_object, parent_to_children_target, parent_field
    )

    def helper(
        source_node: int, target_node: int
    ) -> tuple[int, ExtendedOperation]:
        source_obj = hash_to_object[source_node]
        target_obj = hash_to_object[target_node]

        min_distance, min_operation = recursive_edit_distance(
            source_obj, target_obj, symbol_table
        )
        # Optimization idea:
        # compute the depth and only evaluate the smallest against the childs of the other?

        # Comparing the target to a child of the source node
        for child in parent_to_children_source[source_node]:
            # The source_obj is Trimmed
            # operation = Prune(
            #    KeyValue(parent_field[child]), source_obj, hash_to_object[child]
            # )
            # Computing the distance to the child
            # Only allow one Prune operation
            # child_distance, child_operation = helper(child, target_node)
            child_distance, child_operation = recursive_edit_distance(
                hash_to_object[child], hash_to_object[target_node], symbol_table
            )

            operation = Prune(parent_field[child], child_operation)
            # Trimming then transforming
            distance = (
                source_obj.bit_length()
                - hash_to_object[child].bit_length()
                + child_distance
            )

            if distance < min_distance:
                min_distance, min_operation = (
                    distance,
                    operation,
                )
        for child in parent_to_children_target[target_node]:
            # The child is Attached to the parent
            # operation = Graft(
            #     KeyValue(parent_field[child]), target_obj, hash_to_object[child]
            # )
            # Computing the distance to the child
            # child_distance, child_operation = helper(source_node, child)
            # Only allow one Graft operation
            child_distance, child_operation = recursive_edit_distance(
                hash_to_object[source_node], hash_to_object[child], symbol_table
            )
            operation = Graft(
                parent_field[child], target_obj, child_operation
            )  # transformaing then attaching
            distance = (
                target_obj.bit_length()
                - hash_to_object[child].bit_length()
                + child_distance
            )

            if distance < min_distance:
                min_distance, min_operation = (
                    distance,
                    operation,
                )

        return min_distance, min_operation

    return helper(cached_hash(source), cached_hash(target))


@cache
def recursive_edit_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = tuple(),
    filter_identities=True,
) -> tuple[int, Operation]:
    """
    Args
        source: The source tree, a BitLengthAware object or None.
        target: The target tree, a BitLengthAware object or None.
        symbol_table: A sequence of BitLengthAware objects representing the symbol table, only if source or target can contaons resolvables.
        filter_identites: Wether to simplify the operations structure by filtering away identities

    Returns:
        A tuple of (distance, operations), where distance is the minimal edit cost (in bits),
        and operations is a tuple of ExtendedOperation instances detailing the transformation.
    """

    # Handle the case where the element to compare are reference to a lookup table, possibly with parameters
    # You don't want to compare them literally if they don't point to the same element
    # If both are references then you compute the distance of the resolution of the source to the resolution of the target
    source_resolvable = is_resolvable(source)
    target_resolvable = is_resolvable(target)

    if source_resolvable and target_resolvable:
        if eq_ref(source, target):
            pass  # Same reference, continue with normal comparison
        else:
            d, op = recursive_edit_distance(
                resolve(source, symbol_table),
                resolve(target, symbol_table),
                symbol_table,
                filter_identities,
            )
            return d, Resolve(KeyValue(None), op)  # type: ignore[return-value]
    elif source_resolvable:
        return recursive_edit_distance(
            resolve(source, symbol_table),
            target,
            symbol_table,
            filter_identities,
        )
    elif target_resolvable:
        return recursive_edit_distance(
            source,
            resolve(target, symbol_table),
            symbol_table,
            filter_identities,
        )

    dp = {}

    def helper(
        a: BitLengthAware, b: BitLengthAware, key: KeyValue = KeyValue(None)
    ) -> tuple[int, Operation]:
        ha = cached_hash(a)
        hb = cached_hash(b)
        if (ha, hb) in dp:
            return dp[(ha, hb)]

        if ha == hb:
            # result = (0, [Identity(key, a)])
            result = (0, Identity(key))
        elif (
            not (is_dataclass(a) and is_dataclass(b))
            or isinstance(a, Primitive)
            or isinstance(b, Primitive)
            or type(a) is not type(b)
        ):
            cost = compute_bit_length(a) + compute_bit_length(b)
            result = (cost, Substitute(key, b))
        else:
            total_distance = 0
            operations = set()

            for field in sorted(
                set(fields(a)) | set(fields(b)), key=lambda x: x.name
            ):
                try:
                    a_field = getattr(a, field.name)
                    b_field = getattr(b, field.name)
                except AttributeError as e:
                    raise AttributeError(
                        f"Field {field.name} not found in {a} or {b}"
                    ) from e

                if isinstance(a_field, tuple):
                    if not isinstance(b_field, tuple):
                        raise TypeError(
                            f"{a_field} is a tuple, but not {b_field}"
                        )
                    dist, ops = sequence_edit_distance(
                        a_field, b_field, helper, compute_bit_length, field.name
                    )
                elif isinstance(a_field, frozenset):
                    if not isinstance(b_field, frozenset):
                        raise TypeError(
                            f"{a_field} is a frozenset, but not {b_field}"
                        )
                    dist, ops = set_edit_distance(
                        a_field, b_field, helper, compute_bit_length, field.name
                    )
                elif isinstance(a_field, BitLengthAware):
                    if not isinstance(b_field, BitLengthAware):
                        raise TypeError(
                            f"{a_field} is a BitLengthAware, but not {b_field}"
                        )
                    dist, ops = helper(a_field, b_field, KeyValue(field.name))
                    ops = {ops}
                else:
                    raise TypeError(f"Unknown type: {a_field} and {b_field}")
                if filter_identities:
                    ops = [op for op in ops if not isinstance(op, Identity)]
                operations.update(ops)
                total_distance += dist
            inner = Inner(key, frozenset(operations))
            # substitution = Substitute(key, b)
            # # Arbitrage between Inner and Substitution
            # transformation = (
            #     inner
            #     if total_distance <= a.bit_length() + b.bit_length()
            #     else substitution
            # ) # condition always true
            result = (total_distance, inner)

        dp[(ha, hb)] = result
        return result

    distance, transformation = helper(source, target)
    return distance, transformation


def edit_distance(
    a: str, b: str, reverse: bool = True
) -> tuple[int, Sequence[Operation]]:
    """
    Computes the edit distance between two strings.

    Args:
        a: First string.
        b: Second string.
        reverse: If False, reverses the order of transformations.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    distance, transformations = sequence_edit_distance(
        a,
        b,
        distance_func=lambda x, y: (0, Identity(KeyValue(None)))
        if x == y
        else (2, Substitute(KeyValue(None), y)),
        len_func=lambda x: 1,
    )
    if not reverse:
        tuple(reversed(transformations))
    return distance, transformations


def apply_transformation(
    source: BitLengthAware, operation: ExtendedOperation
) -> BitLengthAware:
    """
    Applies a sequence of transformations to transform the source into the destination.

    Args:
        source: The source BitLengthAware object.
        transformations: Tuple of operations (e.g., Identity, Substitute, Prune, etc.).

    Returns:
        The transformed BitLengthAware object.
    """

    match operation.key.value:
        case None:
            match operation:
                case Identity(_):
                    return source
                case Substitute(_, after):
                    return after
                case Inner(_, operations) if isinstance(source, Primitive):
                    for op in operations:
                        if isinstance(op, Substitute) and op.key.value is None:
                            return op.after  # Apply the substitution
                    print(
                        f"No applicable Substitute found in Inner for Primitive {source}, operation ignored"
                    )
                    return source
                case Inner(_, operations) if not is_dataclass(
                    source
                ) or isinstance(source, Primitive):
                    print(
                        f"Inner operation on a non tree node: {source}, operation ignored"
                    )
                    return source
                case Inner(_, operations):
                    result = source
                    for op in operations:
                        result = apply_transformation(result, op)
                    return result
                case _:
                    print(
                        f"Invalid operation for None key: {operation}, operation ignored"
                    )
                    return source
        case str() as field_name:
            if not is_dataclass(source) or isinstance(source, Primitive):
                print(
                    f"Operation on specific field on non tree node: {source}, operation ignored"
                )
                return source
            else:
                if isinstance(operation, Graft):
                    parent_field = getattr(operation.parent, field_name)
                    return replace(
                        operation.parent,
                        **{
                            field_name: apply_transformation(
                                parent_field, operation.first
                            )
                        },
                    )
                source_field = getattr(source, field_name)
                match operation:
                    case Add(_, value) | Delete(_, value) if not isinstance(
                        source_field, frozenset
                    ):
                        print(
                            f"Trying to add or delete element to a non frozen set field: {source_field} with a string key: {field_name}, operation ignored"
                        )
                        return source
                    case (
                        Identity() | Substitute() | Inner() | Prune() | Graft()
                    ) if not isinstance(source_field, BitLengthAware):
                        print(
                            f"Invalid operation on a non BitLengthAware field: {source_field} with a string key: {field_name}, operation ignored"
                        )
                        return source
                    case Identity(_):
                        return source
                    case Substitute(_, after):
                        return replace(source, **{field_name: after})
                    case Inner(_, ops):
                        assert isinstance(source_field, BitLengthAware)
                        return replace(
                            source,
                            **{
                                field_name: apply_transformation(
                                    source_field, Inner(KeyValue(None), ops)
                                )
                            },
                        )
                    case Prune(_, then):
                        assert isinstance(source_field, BitLengthAware)
                        return apply_transformation(source_field, then)
                    case Add(_, value):
                        assert isinstance(source_field, frozenset)
                        return replace(
                            source,
                            **{field_name: source_field.union({value})},
                        )
                    case Delete(_, value):
                        assert isinstance(source_field, frozenset)
                        return replace(
                            source,
                            **{field_name: source_field.difference({value})},
                        )
                    case _:
                        print(
                            f"Invalid operation for string key: {operation}, operation ignored"
                        )
                        return source
        case (str() as field_name, int() as index):
            if not is_dataclass(source) or isinstance(source, Primitive):
                print(
                    f"Operation on specific field on non tree node: {source}, operation ignored"
                )
                return source
            else:
                if isinstance(operation, Graft):
                    parent_field = getattr(operation.parent, field_name)
                    return replace(
                        operation.parent,
                        **{
                            field_name: parent_field[:index]
                            + (apply_transformation(source, operation.first),)
                            + parent_field[index + 1 :]
                        },
                    )
                source_field = getattr(source, field_name)
                assert isinstance(source_field, tuple)
                match operation:
                    case Identity(_):
                        return source
                    case Substitute(_, after):
                        return replace(
                            source,
                            **{
                                field_name: source_field[:index]
                                + (after,)
                                + source_field[index + 1 :]
                            },
                        )
                    case Inner(_, ops):
                        return replace(
                            source,
                            **{
                                field_name: source_field[:index]
                                + (
                                    apply_transformation(
                                        source_field[index],
                                        Inner(KeyValue(None), ops),
                                    ),
                                )
                                + source_field[index + 1 :]
                            },
                        )
                    case Add(
                        _, value
                    ):  # Migbt add a warning if the length doesn't match? Or remove tuple keys for addition altogether?
                        return replace(
                            source,
                            **{field_name: source_field + (value,)},
                        )
                    case Delete(_, value) if (
                        len(source_field) == index - 1
                        or source_field[-1] == value
                    ):
                        return replace(
                            source, **{field_name: source_field[:-1]}
                        )
                    case Delete(_, value):
                        print(
                            "Delete on out of bound or non matching elements on tuple, operation ignored"
                        )
                        return source
                    case Prune(_, then):
                        assert isinstance(source_field[index], BitLengthAware)
                        return apply_transformation(source_field[index], then)
                    case _:
                        print(
                            f"Invalid operation for tuple key: {operation}, operation ignored"
                        )
                        return source
        case _:
            print(
                f"Invalid key {operation.key.value} for {operation}, operation ignored"
            )
            return source


### Test Functions


def test_edit_distance():
    """Tests the edit distance computation for strings."""
    a = "kitten"
    b = "sitting"

    a_tuple = tuple(CharValue(c) for c in a)
    b_tuple = tuple(CharValue(c) for c in b)

    distance, transformation = sequence_edit_distance(
        a_tuple,
        b_tuple,
        distance_func=lambda x, y: (0, Identity(KeyValue(None)))
        if x == y
        else (2, Substitute(KeyValue(None), y)),
        len_func=lambda x: 1,
    )
    print(f"Distance: {distance}")
    print(
        f"Transformation bit length: {sum(op.bit_length() for op in transformation)}"
    )
    print(f"Distance between {a} and {b}: {distance}")
    print(f"Steps: {len(transformation)}")
    for transformation in transformation:
        print(str(transformation))


def test_set_edit_distance():
    """Tests the edit distance computation for sets."""
    source_set = frozenset({CharValue("a"), CharValue("b")})
    target_set = frozenset({CharValue("a"), CharValue("c")})
    distance, transformation = set_edit_distance(
        source_set,
        target_set,
        distance_func=lambda x, y: (0, Identity(KeyValue(None)))
        if x == y
        else (2, Substitute(KeyValue(None), y)),
        len_func=lambda x: 1,
    )
    print(f"Distance: {distance}")
    print(f"Transformation: {transformation}")
    print(
        f"Transformation bit length: {sum(op.bit_length() for op in transformation)}"
    )
    print(f"Distance between {source_set} and {target_set}: {distance}")
    print("Transformations:")
    for operation in transformation:
        print(str(operation))


@dataclass(frozen=True)
class MockValue(Primitive):
    value: int

    def bit_length(self) -> int:
        return 1


@dataclass(frozen=True)
class CharValue(Primitive):
    value: str

    def bit_length(self) -> int:
        return 1


@dataclass(frozen=True)
class TreeNode(BitLengthAware, RoseNode[MockValue]):
    children: "tuple[TreeNode, ...]" = field(default_factory=tuple)

    def bit_length(self) -> int:
        """Return the size of the subtree (number of nodes)."""
        return self.value.bit_length() + sum(
            child.bit_length() for child in self.children
        )

    def __str__(self):
        return f"{self.value} -> ({'|'.join(str(child) for child in self.children)})"


def test_recursive_edit_distance():
    # Create primitive values
    prim1 = MockValue(1)
    prim2 = MockValue(2)
    prim3 = MockValue(3)
    prim4 = MockValue(4)
    prim5 = MockValue(5)

    # Create leaf nodes
    leaf1 = TreeNode(prim1)
    leaf2 = TreeNode(prim2)
    leaf3 = TreeNode(prim3)
    leaf4 = TreeNode(prim4)

    # Create test trees
    root1 = TreeNode(prim1, (leaf1, leaf2))  # Tree: 1 -> (1, 2)
    root2 = TreeNode(
        prim1, (leaf1, leaf2)
    )  # Tree: 1 -> (1, 2), identical to root1
    root3 = TreeNode(prim1, (leaf1, leaf3))  # Tree: 1 -> (1, 3)
    root4 = TreeNode(prim1, (leaf1, leaf2, leaf4))  # Tree: 1 -> (1, 2, 4)
    root5 = TreeNode(prim5)  # Tree: 5

    # **Test Case 1: Identical Trees**
    # distance, transformations = recursive_edit_distance(root1, root2)
    distance, transformation = recursive_edit_distance(root1, root2)
    assert distance == 0, "Distance should be 0 for identical trees"
    assert isinstance(transformation, Identity), (
        "Transformation should be Identity"
    )
    print("Test 1 (Identical trees): Passed")

    # **Test Case 2: One Leaf Value Changed**
    distance, transformation = recursive_edit_distance(root1, root3)
    print(f"distance: {distance}, transformations: {transformation}")
    assert distance == 2, "Distance should be 2 (substitute leaf2 with leaf3)"
    assert any(
        isinstance(operation, Substitute)
        for operation in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Substitute"
    print("Test 2 (One leaf changed): Passed")

    # **Test Case 3: Added Node**
    distance, transformation = recursive_edit_distance(root1, root4)
    assert distance == 1, "Distance should be 1 (add leaf4)"
    print(f"transformation: {transformation}")
    assert any(
        isinstance(op, Add)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Add"
    print("Test 3 (Added node): Passed")

    # **Test Case 4: Completely Different Trees**
    distance, transformation = recursive_edit_distance(root1, root5)
    assert distance == 4, (
        "Distance should be 4 (substitute value, delete two children)"
    )
    assert any(
        isinstance(op, Substitute)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Substitute"
    assert any(
        isinstance(op, Delete)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Delete"
    print("Test 4 (Different trees): Passed")

    print("\nAll tests completed successfully!")
    print("Sample transformations (root1 to root3):")
    _, sample_transforms = recursive_edit_distance(root1, root3)
    print(f"  {sample_transforms}")


def test_extended_edit_distance():
    """
    Tests the extended_edit_distance function, focusing on Prune and Graft operations.
    """

    # Create leaf nodes
    leaf_E = TreeNode(MockValue(3))  # Node C with value 3
    leaf_D = TreeNode(MockValue(3))  # Node C with value 3
    leaf_C = TreeNode(MockValue(2))  # Node B with value 2

    node_B = TreeNode(MockValue(0), (leaf_C, leaf_D))

    # Create parent node A with children B and C
    node_A = TreeNode(MockValue(1), (node_B, leaf_E))  # A -> (B, C)

    # Test 1: Prune operation
    # Source: A -> (B, E), Target: B
    print("Testing Prune operation:")
    distance, transformation = extended_edit_distance(node_A, node_B, tuple())
    assert transformation is not None, "The trasnformation should not be None"
    print(f"Distance: {distance}")
    print(f"Transformation {transformation}:")
    assert distance == 2, f"Expected distance 2 for Prune, got {distance}"
    assert any(
        isinstance(op, Prune)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Transformations should include Prune"
    print("Prune test passed.\n")

    # Test 2: Graft operation
    # Source: B, Target: A -> (B, E)
    print("Testing Graft operation:")
    distance, transformation = extended_edit_distance(node_B, node_A, tuple())
    assert transformation is not None, "The trasnformation should not be None"
    print(f"Distance: {distance}")
    print(f"Transformation: {transformation}")
    assert distance == 2, f"Expected distance 2 for Graft, got {distance}"
    assert any(
        isinstance(op, Graft)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Transformations should include Graft"
    print("Graft test passed.\n")

    print("All extended edit distance tests passed successfully!")


def test_apply_transformations():
    """
    Tests the application of transformations to ensure source objects are correctly
    transformed into target objects.
    """
    print("Starting transformation application tests...\n")

    # Wrapper for tuples
    @dataclass(frozen=True)
    class TupleWrapper(BitLengthAware):
        items: tuple[MockValue, ...]

        def bit_length(self) -> int:
            return sum(item.bit_length() for item in self.items)

        def __str__(self) -> str:
            return (
                f"TupleWrapper({', '.join(str(item) for item in self.items)})"
            )

    # **Test Case 1: Simple Primitive Substitution**
    source_1 = MockValue(1)
    target_1 = MockValue(2)
    _, transform_1 = recursive_edit_distance(source_1, target_1)
    print(f" Transformation: {transform_1}")
    result_1 = apply_transformation(source_1, transform_1)
    assert result_1 == target_1, f"Test 1 failed: {result_1} != {target_1}"
    print("Test 1 (Primitive Substitution): Passed")

    # Test Case 2: Tuple
    source_2 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3)))
    target_2 = TupleWrapper((MockValue(1), MockValue(4), MockValue(3)))
    _, transform_2 = recursive_edit_distance(source_2, target_2)
    print(f" Transformation: {transform_2}")
    result_2 = apply_transformation(source_2, transform_2)
    print(f"result_2: {[result_2]}")
    assert result_2 == target_2, f"Test 2 failed: {result_2} != {target_2}"
    print("Test 2 (tuple): Passed")

    # **Test Case 3: Dataclass with Tuple Field**
    source_3 = TreeNode(
        MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(3)))
    )
    target_3 = TreeNode(
        MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(4)))
    )
    _, transform_3 = recursive_edit_distance(source_3, target_3)
    print(f" Transformation: {transform_3}")
    result_3 = apply_transformation(source_3, transform_3)
    assert result_3 == target_3, f"Test 3 failed: {result_3} != {target_3}"
    print("Test 3 (Dataclass with Tuple): Passed")

    # **Test Case 4: Prune Operation**
    leaf_E = TreeNode(MockValue(3))
    leaf_D = TreeNode(MockValue(3))
    leaf_C = TreeNode(MockValue(2))
    node_B = TreeNode(MockValue(0), (leaf_C, leaf_D))
    node_A = TreeNode(MockValue(1), (node_B, leaf_E))
    source_4 = node_A
    target_4 = node_B
    _, transform_4 = extended_edit_distance(source_4, target_4, tuple())
    assert transform_4 is not None, "The trasnformation should not be None"
    result_4 = apply_transformation(source_4, transform_4)
    assert result_4 == target_4, f"Test 4 failed: {result_4} != {target_4}"
    print("Test 4 (Prune Operation): Passed")

    # **Test Case 5: Graft Operation**
    source_5 = node_B
    target_5 = node_A
    _, transform_5 = extended_edit_distance(source_5, target_5, tuple())
    assert transform_5 is not None, "The trasnformation should not be None"
    result_5 = apply_transformation(source_5, transform_5)
    assert result_5 == target_5, f"Test 5 failed: {result_5} != {target_5}"
    print("Test 5 (Graft Operation): Passed")

    print("\nAll transformation application tests passed successfully!")


if __name__ == "__main__":
    test_edit_distance()
    test_set_edit_distance()
    test_recursive_edit_distance()
    test_extended_edit_distance()
    test_apply_transformations()
