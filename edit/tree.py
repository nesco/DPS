"""
Tree edit distance algorithms.

Cost models:
- Edit: delete costs bit_length(element), prune costs bit_length difference
- MDL: delete costs log2(collection_size), prune costs log2(path_space)
"""

import math
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Sequence

from kolmogorov_tree.resolution import eq_ref, is_resolvable, resolve
from localtypes import BitLengthAware, KeyValue, Primitive, ensure_all_instances

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
)
from .sequence import sequence_edit_distance
from .sets import set_edit_distance


def compute_bit_length(obj: Any) -> int:
    """Compute bit length for any object, including collections."""
    if isinstance(obj, BitLengthAware):
        return obj.bit_length()
    elif isinstance(obj, (set, frozenset, tuple, list)):
        return sum(compute_bit_length(elem) for elem in obj)
    else:
        return 1


def collect_links(
    bla: BitLengthAware,
    hash_to_object: dict[int, BitLengthAware],
    parent_to_children: defaultdict[int, set[int]],
    parent_field: dict[int, KeyValue],
):
    """Collect topological structure of a BitLengthAware tree via preorder traversal."""
    bla_hash = hash(bla)
    hash_to_object[bla_hash] = bla

    if is_dataclass(bla) and not isinstance(bla, Primitive):
        for field in fields(bla):
            attr = getattr(bla, field.name)
            if isinstance(attr, BitLengthAware):
                field_hash = hash(attr)
                parent_to_children[bla_hash].add(field_hash)
                parent_field[field_hash] = KeyValue(field.name)
                collect_links(attr, hash_to_object, parent_to_children, parent_field)
            if isinstance(attr, tuple):
                attr = ensure_all_instances(attr, BitLengthAware)
                for i, elem in enumerate(attr):
                    field_hash = hash(elem)
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
                    field_hash = hash(elem)
                    parent_to_children[bla_hash].add(field_hash)
                    parent_field[field_hash] = KeyValue(field.name)
                    collect_links(
                        elem,
                        hash_to_object,
                        parent_to_children,
                        parent_field,
                    )


def collect_all_descendants(
    node_hash: int,
    parent_to_children: defaultdict[int, set[int]],
    parent_field: dict[int, KeyValue],
) -> list[tuple[int, list[KeyValue]]]:
    """
    Collect all descendants of a node with their key paths.

    Returns list of (descendant_hash, [key1, key2, ...]) tuples,
    where the key path represents the navigation from node to descendant.
    """
    descendants: list[tuple[int, list[KeyValue]]] = []

    def recurse(current: int, path: list[KeyValue]):
        for child in parent_to_children[current]:
            child_path = path + [parent_field[child]]
            descendants.append((child, child_path))
            recurse(child, child_path)

    recurse(node_hash, [])
    return descendants


def build_nested_prune(
    key_path: list[KeyValue], inner_op: ExtendedOperation
) -> ExtendedOperation:
    """
    Build a nested Prune operation from a key path.

    For path [k1, k2, k3] and inner_op, builds:
        Prune(k1, Prune(k2, Prune(k3, inner_op)))
    """
    result = inner_op
    for key in reversed(key_path):
        result = Prune(key, result)
    return result


def build_nested_graft(
    key_path: list[KeyValue],
    ancestors: list[BitLengthAware],
    inner_op: ExtendedOperation,
) -> ExtendedOperation:
    """
    Build a nested Graft operation from a key path and ancestor chain.

    For path [k1, k2, k3], ancestors [a1, a2, a3] and inner_op, builds:
        Graft(k1, a1, Graft(k2, a2, Graft(k3, a3, inner_op)))
    """
    result = inner_op
    for key, ancestor in zip(reversed(key_path), reversed(ancestors)):
        result = Graft(key, ancestor, result)
    return result


def extended_edit_distance(
    source: BitLengthAware | None,
    target: BitLengthAware | None,
    symbol_table: Sequence[BitLengthAware],
    metric: Literal["edit", "mdl"] = "edit",
) -> tuple[int, ExtendedOperation | None]:
    """
    Computes extended edit distance with Prune and Graft operations.

    Args:
        source: Source tree (or None).
        target: Target tree (or None).
        symbol_table: Symbols for resolving references.
        metric: "edit" for transformation distance, "mdl" for information distance.

    Returns:
        Tuple of (distance, operation).
    """
    # Handle resolvable nodes (SymbolNode, NestedNode)
    source_resolvable = source is not None and is_resolvable(source)
    target_resolvable = target is not None and is_resolvable(target)

    if source_resolvable and target_resolvable:
        assert source is not None and target is not None
        if eq_ref(source, target):
            pass  # Same reference, continue with normal comparison
        else:
            d, op = extended_edit_distance(
                resolve(source, symbol_table),
                resolve(target, symbol_table),
                symbol_table,
                metric,
            )
            return d, Resolve(KeyValue(None), op) if op is not None else (d, None)  # type: ignore[return-value]
    elif source_resolvable:
        assert source is not None
        return extended_edit_distance(
            resolve(source, symbol_table),
            target,
            symbol_table,
            metric,
        )
    elif target_resolvable:
        assert target is not None
        return extended_edit_distance(
            source,
            resolve(target, symbol_table),
            symbol_table,
            metric,
        )

    if source is None and target is None:
        return 0, None

    if source is None:
        assert target is not None
        return (target.bit_length(), Add(KeyValue(None), target))

    if target is None:
        # For delete at tree level, use bit_length for both metrics
        # (MDL log2 cost applies to collection elements, not root deletion)
        return (compute_bit_length(source), Delete(KeyValue(None), source))

    # Collect the object topological structure
    hash_to_object: dict[int, BitLengthAware] = {}
    parent_to_children_source: defaultdict[int, set[int]] = defaultdict(set)
    parent_to_children_target: defaultdict[int, set[int]] = defaultdict(set)
    parent_field: dict[int, KeyValue] = {}
    child_to_parent: dict[int, int] = {}

    collect_links(source, hash_to_object, parent_to_children_source, parent_field)
    collect_links(target, hash_to_object, parent_to_children_target, parent_field)

    # Build child -> parent mapping for ancestor chain construction
    for parent, children in parent_to_children_target.items():
        for child in children:
            child_to_parent[child] = parent

    def get_ancestor_chain(node: int, root: int) -> list[BitLengthAware]:
        """Get list of ancestors from root to node's parent (for Graft)."""
        ancestors = []
        current = node
        while current in child_to_parent and current != root:
            parent = child_to_parent[current]
            ancestors.append(hash_to_object[parent])
            current = parent
        return list(reversed(ancestors))

    def helper(source_node: int, target_node: int) -> tuple[int, ExtendedOperation]:
        source_obj = hash_to_object[source_node]
        target_obj = hash_to_object[target_node]

        min_distance, min_operation = recursive_edit_distance(
            source_obj, target_obj, symbol_table, True, metric
        )

        # Prune: compare target to ALL descendants of source
        source_descendants = collect_all_descendants(
            source_node, parent_to_children_source, parent_field
        )

        for desc_hash, key_path in source_descendants:
            desc_obj = hash_to_object[desc_hash]
            desc_distance, desc_operation = recursive_edit_distance(
                desc_obj, target_obj, symbol_table, True, metric
            )

            # Prune cost depends on metric
            if metric == "mdl":
                # MDL: cost to identify which path = log2(path_space)
                num_descendants = len(source_descendants)
                prune_cost = (
                    math.ceil(math.log2(num_descendants)) if num_descendants > 1 else 0
                )
            else:
                # Edit: cost is the structure being removed
                prune_cost = compute_bit_length(source_obj) - compute_bit_length(
                    desc_obj
                )

            distance = prune_cost + desc_distance

            if distance < min_distance:
                operation = build_nested_prune(key_path, desc_operation)
                min_distance, min_operation = distance, operation

        # Graft: compare source to ALL descendants of target
        target_descendants = collect_all_descendants(
            target_node, parent_to_children_target, parent_field
        )

        for desc_hash, key_path in target_descendants:
            desc_obj = hash_to_object[desc_hash]
            desc_distance, desc_operation = recursive_edit_distance(
                source_obj, desc_obj, symbol_table, True, metric
            )

            # Graft cost: adding ancestor structure (same for both metrics)
            graft_cost = target_obj.bit_length() - desc_obj.bit_length()
            distance = graft_cost + desc_distance

            if distance < min_distance:
                ancestors = get_ancestor_chain(desc_hash, target_node)
                all_ancestors = [target_obj] + ancestors
                operation = build_nested_graft(key_path, all_ancestors, desc_operation)
                min_distance, min_operation = distance, operation

        return min_distance, min_operation

    return helper(hash(source), hash(target))


def recursive_edit_distance(
    source: BitLengthAware,
    target: BitLengthAware,
    symbol_table: Sequence[BitLengthAware] = (),
    filter_identities: bool = True,
    metric: Literal["edit", "mdl"] = "edit",
) -> tuple[int, Operation]:
    """
    Compute recursive edit distance between two BitLengthAware objects.

    Args:
        source: Source tree.
        target: Target tree.
        symbol_table: Symbols for resolving references.
        filter_identities: Whether to filter Identity operations from results.
        metric: "edit" for transformation distance, "mdl" for information distance.

    Returns:
        Tuple of (distance, operation).
    """
    # Handle resolvable references
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
                metric,
            )
            return d, Resolve(KeyValue(None), op)  # type: ignore[return-value]
    elif source_resolvable:
        return recursive_edit_distance(
            resolve(source, symbol_table),
            target,
            symbol_table,
            filter_identities,
            metric,
        )
    elif target_resolvable:
        return recursive_edit_distance(
            source,
            resolve(target, symbol_table),
            symbol_table,
            filter_identities,
            metric,
        )

    dp: dict[tuple[int, int], tuple[int, Operation]] = {}

    def helper(
        a: BitLengthAware, b: BitLengthAware, key: KeyValue = KeyValue(None)
    ) -> tuple[int, Operation]:
        ha = hash(a)
        hb = hash(b)
        if (ha, hb) in dp:
            return dp[(ha, hb)]

        if ha == hb:
            result = (0, Identity(key))
        elif (
            not (is_dataclass(a) and is_dataclass(b))
            or isinstance(a, Primitive)
            or isinstance(b, Primitive)
            or type(a) is not type(b)
        ):
            # Different types or primitives: substitute costs describing the new value
            cost = compute_bit_length(b)
            result = (cost, Substitute(key, b))
        else:
            total_distance = 0
            operations: set[Operation] = set()

            for field in sorted(set(fields(a)) | set(fields(b)), key=lambda x: x.name):
                try:
                    a_field = getattr(a, field.name)
                    b_field = getattr(b, field.name)
                except AttributeError as e:
                    raise AttributeError(
                        f"Field {field.name} not found in {a} or {b}"
                    ) from e

                if isinstance(a_field, tuple):
                    if not isinstance(b_field, tuple):
                        raise TypeError(f"{a_field} is a tuple, but not {b_field}")
                    dist, ops = sequence_edit_distance(
                        a_field,
                        b_field,
                        helper,
                        compute_bit_length,
                        field.name,
                        metric,
                    )
                elif isinstance(a_field, frozenset):
                    if not isinstance(b_field, frozenset):
                        raise TypeError(f"{a_field} is a frozenset, but not {b_field}")
                    dist, ops = set_edit_distance(
                        a_field,
                        b_field,
                        helper,
                        compute_bit_length,
                        field.name,
                        metric,
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

            result = (total_distance, Inner(key, frozenset(operations)))

        dp[(ha, hb)] = result
        return result

    return helper(source, target)
