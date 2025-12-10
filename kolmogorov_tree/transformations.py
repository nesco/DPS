"""
Tree transformation and compression utilities.

Key functions:
    Mapping:
        postmap(node, f)  - Apply f bottom-up (children first, then parent)
        premap(node, f)   - Apply f top-down (parent first, then children)

    Value transforms:
        shift(node, k)    - Shift all Alphabet values by k
        reverse_node(node) - Reverse sequence order recursively

    Compression:
        encode_run_length(nodes)      - RLE for consecutive identical nodes
        find_repeating_pattern(nodes) - Find best repeating subsequence
        factorize_tuple(node)         - Compress ProductNode/SumNode patterns
        get_iterator(nodes)           - Detect arithmetic sequences in SumNode

    Construction:
        construct_product_node(nodes) - Build optimized ProductNode
        iterable_to_product(nodes)    - Convert to ProductNode or single node
        iterable_to_sum(nodes)        - Convert to SumNode or single node
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Callable, Iterable, Sequence, cast

from utils.tree_functionals import postorder_map, preorder_map

from kolmogorov_tree.nodes import (
    KNode,
    NestedNode,
    NoneValue,
    PrimitiveNode,
    ProductNode,
    RepeatNode,
    RootNode,
    SumNode,
    SymbolNode,
    VariableNode,
)
from kolmogorov_tree.primitives import (
    Alphabet,
    CountValue,
    MoveValue,
    T,
)
from kolmogorov_tree.traversal import children


def shift_f(node: KNode[T], k: int) -> KNode[T]:
    """Shifts a single node's Alphabet value by k (helper for shift)."""
    if isinstance(node, PrimitiveNode) and isinstance(node.value, Alphabet):
        return PrimitiveNode[T](node.value.shift(k))
    return node


def reconstruct_knode(
    knode: KNode[T], new_children: Sequence[KNode[T]], factorize: bool = False
) -> KNode[T]:
    """Rebuilds a node with the same type/metadata but new children."""
    match knode:
        case SumNode():
            result = SumNode(frozenset(new_children))
            return factorize_tuple(result) if factorize else result
        case ProductNode():
            result = ProductNode(tuple(new_children))
            return factorize_tuple(result) if factorize else result
        case RepeatNode(_, count):
            return RepeatNode(new_children[0], count)
        case NestedNode(index, _, count):
            return NestedNode(index, new_children[0], count)
        case RootNode(_, position, colors):
            child = new_children[0] if new_children else NoneValue()
            return RootNode(child, position, colors)
        case SymbolNode(index, parameters):
            new_params = list(parameters)
            for i, child in enumerate(new_children):
                new_params[i] = child
            return SymbolNode(index, tuple(new_params))
        case _:
            return knode


def postmap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode[T]:
    """Applies f to each node bottom-up (children transformed before parent)."""
    return postorder_map(
        knode, f, children, lambda n, kids: reconstruct_knode(n, kids, factorize)
    )


def premap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode:
    """Applies f to each node top-down (parent transformed before children)."""
    return preorder_map(
        knode, f, children, lambda n, kids: reconstruct_knode(n, kids, factorize)
    )


def shift(node: KNode[T], k: int) -> KNode[T]:
    """Shifts all Alphabet values in the tree by k (with modular wraparound)."""
    return postmap(node, lambda n: shift_f(n, k))


def reverse_node(knode: KNode[T]) -> KNode[T]:
    """Recursively reverses sequence order in ProductNodes and parameters."""
    match knode:
        case ProductNode(children):
            return ProductNode(tuple(reverse_node(c) for c in reversed(children)))
        case SumNode(children):
            return SumNode(frozenset(reverse_node(c) for c in children))
        case RepeatNode(node, count):
            return RepeatNode(reverse_node(node), count)
        case NestedNode(index, node, count):
            return NestedNode(index, reverse_node(node), count)
        case SymbolNode(index, parameters):
            reversed_params = tuple(
                reverse_node(p) if isinstance(p, KNode) else p
                for p in reversed(parameters)
            )
            return SymbolNode(index, reversed_params)
        case RootNode(node, position, colors):
            inner = NoneValue() if isinstance(node, NoneValue) else reverse_node(node)
            return RootNode(inner, position, colors)
        case _:
            return knode


def encode_run_length(
    primitives: Iterable[PrimitiveNode[T]],
) -> ProductNode[T] | RepeatNode[T] | PrimitiveNode[T]:
    """Compresses consecutive identical PrimitiveNodes into RepeatNodes (runs of 3+)."""
    result: list[KNode[T]] = []
    iterator = iter(primitives)

    try:
        current = next(iterator)
    except StopIteration:
        return ProductNode(())

    run_length = 1
    for node in iterator:
        if node == current:
            run_length += 1
        else:
            if run_length >= 3:
                result.append(RepeatNode(current, CountValue(run_length)))
            else:
                result.extend([current] * run_length)
            current = node
            run_length = 1

    if run_length >= 3:
        result.append(RepeatNode(current, CountValue(run_length)))
    else:
        result.extend([current] * run_length)

    if len(result) == 1:
        single = result[0]
        assert isinstance(single, (RepeatNode, PrimitiveNode))
        return single
    return ProductNode(tuple(result))


def find_repeating_pattern(
    nodes: Sequence[KNode[T]], offset: int
) -> tuple[KNode[T] | None, int, bool]:
    """
    Finds the best repeating pattern starting at offset.

    Returns (pattern, count, is_alternating) where:
        - pattern: The repeating unit, or None if no beneficial pattern found
        - count: Positive for normal repeat, negative for alternating with reverse
        - is_alternating: True if pattern alternates with its reverse
    """
    if offset >= len(nodes):
        return None, 0, False

    best: tuple[KNode[T] | None, int, int, bool] = (None, 0, 0, False)
    max_pattern_len = (len(nodes) - offset + 1) // 2

    segment_cache: dict[tuple[int, int], KNode[T]] = {}
    reverse_cache: dict[KNode[T], KNode[T]] = {}

    def get_segment(start: int, end: int) -> KNode[T]:
        if (start, end) not in segment_cache:
            seg = nodes[start:end]
            segment_cache[(start, end)] = (
                iterable_to_product(seg) if len(seg) > 1 else seg[0]
            )
        return segment_cache[(start, end)]

    def get_reversed(node: KNode[T]) -> KNode[T]:
        if node not in reverse_cache:
            reverse_cache[node] = reverse_node(node)
        return reverse_cache[node]

    for pattern_len in range(1, max_pattern_len + 1):
        pattern = get_segment(offset, offset + pattern_len)

        for alternating in [False, True]:
            count = 1
            pos = offset + pattern_len

            while pos + pattern_len <= len(nodes):
                expected = (
                    get_reversed(pattern)
                    if (alternating and count % 2 == 1)
                    else pattern
                )
                actual = get_segment(pos, pos + pattern_len)
                if actual != expected:
                    break
                count += 1
                pos += pattern_len

            if count > 1:
                span = nodes[offset : offset + pattern_len * count]
                original_bits = sum(n.bit_length() for n in span)
                signed_count = -count if alternating else count
                compressed_bits = RepeatNode(
                    pattern, CountValue(signed_count)
                ).bit_length()
                bit_gain = original_bits - compressed_bits

                if bit_gain > best[2]:
                    best = (pattern, count, bit_gain, alternating)

    pattern, count, _, alternating = best
    return pattern, (-count if alternating else count), alternating


def flatten_sum(node: SumNode[T]) -> KNode[T]:
    """Unwraps single-child SumNode unless it's a RepeatNode (iterator encoding)."""
    if len(node.children) != 1:
        return node
    child = next(iter(node.children))
    return node if isinstance(child, RepeatNode) else child


def flatten_product(node: ProductNode[T]) -> KNode[T]:
    """Unwraps single-child ProductNode."""
    if not node.children:
        raise ValueError("ProductNode has no children")
    return node.children[0] if len(node.children) == 1 else node


def get_iterator(knodes: Collection[KNode[T]]) -> frozenset[KNode[T]]:
    """
    Detects arithmetic sequences and encodes as RepeatNode.

    For SumNode children forming a sequence like {0,1,2}, returns
    {RepeatNode(0, 3)} which means "0, shift(0,1), shift(0,2)".
    """
    nodes = frozenset(knodes)
    if len(nodes) < 2:
        return nodes

    is_shiftable = (
        all(
            isinstance(n, PrimitiveNode) and isinstance(n.value, MoveValue)
            for n in nodes
        )
        or all(isinstance(n, RepeatNode) for n in nodes)
        or all(
            isinstance(n, ProductNode)
            and all(
                isinstance(p, PrimitiveNode) and isinstance(p.value, MoveValue)
                for p in n.children
            )
            for n in nodes
        )
    )

    if not is_shiftable:
        return nodes

    n = len(nodes)
    start = next(iter(nodes))

    # Try to form ascending chain by traversing both directions
    visited: set[KNode[T]] = {start}
    current = start
    while (next_node := shift(current, 1)) in nodes and next_node not in visited:
        visited.add(next_node)
        current = next_node

    current = start
    while (prev_node := shift(current, -1)) in nodes and prev_node not in visited:
        visited.add(prev_node)
        current = prev_node
    chain_start = current

    if len(visited) == n:
        return frozenset([RepeatNode(chain_start, CountValue(n))])

    # Try descending chain
    visited = {start}
    current = start
    while (next_node := shift(current, -1)) in nodes and next_node not in visited:
        visited.add(next_node)
        current = next_node

    current = start
    while (prev_node := shift(current, 1)) in nodes and prev_node not in visited:
        visited.add(prev_node)
        current = prev_node
    chain_start = current

    if len(visited) == n:
        return frozenset([RepeatNode(chain_start, CountValue(-n))])

    return nodes


def construct_product_node(nodes: Iterable[KNode[T]]) -> ProductNode[T]:
    """
    Builds an optimized ProductNode with flattening and run-length encoding.

    - Flattens nested ProductNodes
    - Merges consecutive identical primitives into RepeatNodes
    - Merges adjacent RepeatNodes with same base
    """
    flattened: list[KNode[T]] = []
    for node in nodes:
        if isinstance(node, ProductNode):
            flattened.extend(node.children)
        else:
            flattened.append(node)

    result: list[KNode[T]] = []
    primitive_buffer: list[PrimitiveNode[T]] = []

    def flush_primitives() -> None:
        nonlocal primitive_buffer
        if not primitive_buffer:
            return
        if len(primitive_buffer) == 1:
            result.append(primitive_buffer[0])
        else:
            encoded = encode_run_length(primitive_buffer)
            if isinstance(encoded, ProductNode):
                result.extend(encoded.children)
            else:
                result.append(encoded)
        primitive_buffer = []

    def try_merge_repeat(node: RepeatNode[T]) -> None:
        if (
            result
            and isinstance(result[-1], RepeatNode)
            and isinstance(result[-1].count, CountValue)
            and isinstance(node.count, CountValue)
            and result[-1].node == node.node
        ):
            combined = CountValue(result[-1].count.value + node.count.value)
            result[-1] = RepeatNode(node.node, combined)
        else:
            result.append(node)

    for node in flattened:
        if isinstance(node, PrimitiveNode):
            primitive_buffer.append(node)
        elif isinstance(node, RepeatNode):
            flush_primitives()
            try_merge_repeat(node)
        else:
            flush_primitives()
            result.append(node)

    flush_primitives()
    return ProductNode(tuple(result))


def iterable_to_product(iterable: Iterable[KNode[T]]) -> KNode[T]:
    """Converts iterable to ProductNode, or returns single element directly."""
    nodes = list(iterable)
    if not nodes:
        raise ValueError("Cannot create product from empty iterable")
    if len(nodes) == 1:
        return nodes[0]
    return flatten_product(construct_product_node(nodes))


def iterable_to_sum(
    iterable: Iterable[KNode[T]], compress_iterator: bool = False
) -> KNode[T] | None:
    """Converts iterable to SumNode, or returns single element directly."""
    nodes = frozenset(iterable)
    if not nodes:
        return None
    if len(nodes) == 1:
        return next(iter(nodes))
    if compress_iterator:
        return SumNode(get_iterator(nodes))
    return SumNode(nodes)


def factorize_tuple(node: KNode[T]) -> KNode[T]:
    """Compresses ProductNode/SumNode by finding repeating patterns."""
    if isinstance(node, SumNode):
        return flatten_sum(SumNode(get_iterator(node.children)))

    if not isinstance(node, ProductNode) or len(node.children) < 2:
        return node

    children_list = list(node.children)
    result: list[KNode[T]] = []
    i = 0

    while i < len(children_list):
        pattern, count, _ = find_repeating_pattern(children_list, i)
        if pattern is not None and abs(count) > 1:
            result.append(RepeatNode(pattern, CountValue(count)))
            pattern_len = (
                len(pattern.children) if isinstance(pattern, ProductNode) else 1
            )
            i += abs(count) * pattern_len
        else:
            result.append(children_list[i])
            i += 1

    if len(result) == 1 and result[0] == node:
        return node
    return cast(KNode, iterable_to_product(result))


def expand_repeats(node: KNode[T]) -> KNode[T]:
    """Expands all RepeatNodes into explicit ProductNodes."""

    def expand(knode: KNode) -> KNode:
        if isinstance(knode, RepeatNode):
            if isinstance(knode.count, VariableNode):
                raise TypeError("Cannot expand RepeatNode with variable count")
            return ProductNode((knode.node,) * knode.count.value)
        return knode

    return cast(
        SumNode[T] | ProductNode[T] | PrimitiveNode[T],
        postmap(node, expand, factorize=False),
    )


__all__ = [
    "shift_f",
    "reconstruct_knode",
    "postmap",
    "premap",
    "shift",
    "reverse_node",
    "encode_run_length",
    "find_repeating_pattern",
    "flatten_sum",
    "flatten_product",
    "get_iterator",
    "construct_product_node",
    "iterable_to_product",
    "iterable_to_sum",
    "factorize_tuple",
    "expand_repeats",
]
