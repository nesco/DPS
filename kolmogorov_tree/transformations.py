"""
Transformation utilities for the Kolmogorov Tree.

This module provides functions to transform and manipulate KNodes:
- postmap, premap: Map functions over tree structure
- reconstruct_knode: Rebuild nodes with new children
- shift, reverse_node: Transform node values
- encode_run_length, find_repeating_pattern: Compression utilities
- factorize_tuple, flatten_sum, flatten_product: Simplification utilities
- expand_repeats: Decompression utility
- construct_product_node, iterable_to_product, iterable_to_sum: Node construction
- get_iterator: Arithmetic sequence detection
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


# Helper for shift
def shift_f(node: KNode[T], k: int) -> KNode[T]:
    if isinstance(node, PrimitiveNode) and isinstance(node.value, Alphabet):
        shifted_value = node.value.shift(k)
        return PrimitiveNode[T](shifted_value)
    return node


def reconstruct_knode(
    knode: KNode[T], new_children: Sequence[KNode[T]], factorize: bool = False
) -> KNode[T]:
    """Reconstructs a KNode with its original data and new children."""
    match knode:
        case SumNode(_):
            new_node = SumNode(frozenset(new_children))
            if factorize:
                new_node = factorize_tuple(new_node)
            return new_node
        case ProductNode(_):
            new_node = ProductNode(tuple(new_children))
            if factorize:
                new_node = factorize_tuple(new_node)
            return new_node
        case RepeatNode(_, count):
            return RepeatNode(new_children[0], count)
        case NestedNode(index, _, count):
            return NestedNode(index, new_children[0], count)
        case RootNode(_, position, colors):
            if not new_children:
                return RootNode(NoneValue(), position, colors)
            return RootNode(new_children[0], position, colors)
        case SymbolNode(index, parameters):
            new_parameters = list(parameters)
            for i, p in enumerate(new_children):
                new_parameters[i] = p
            return SymbolNode(index, tuple(new_parameters))
        case _:
            return knode  # Leaf nodes remain unchanged


def postmap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode[T]:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.
        factorize: Whether to automatically factorize the TupleNodes of the tree

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    return postorder_map(
        knode,
        f,
        children,
        lambda node, kids: reconstruct_knode(node, kids, factorize),
    )


def premap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode:
    """
    Map a function alongside a KNode. It updates the base node first, then updates the children

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.
        factorize: Whether to automatically factorize the TupleNodes of the tree

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    return preorder_map(
        knode,
        f,
        children,
        lambda node, kids: reconstruct_knode(node, kids, factorize),
    )


def shift(node: KNode[T], k: int) -> KNode[T]:
    """
    Shifts the values of all PrimitiveNodes in the KolmogorovTree that contain Alphabet subclasses by k.

    This function recursively traverses the KolmogorovTree using the postmap function and applies the shift
    operation to any PrimitiveNode that holds a value of a type that is a subclass of Alphabet. The shift
    operation is defined by the `shift` method of the Alphabet subclass.

    Parameters:
    -----------
    node : KNode[T]
        The root node of the KolmogorovTree to shift.
    k : int
        The amount to shift the values by.

    Returns:
    --------
    KNode[T]
        A new KolmogorovTree with the same structure, but with all shiftable values shifted by k.

    Examples:
    ---------
    >>> node = PrimitiveNode(MoveValue(1))
    >>> shifted = shift(node, 1)
    >>> shifted.data  # Assuming MoveValue shifts modulo 8
    2
    >>> node = ProductNode((PrimitiveNode(MoveValue(1)), PrimitiveNode(CountValue(2))))
    >>> shifted = shift(node, 1)
    >>> shifted.children[0].data  # Shifted MoveValue
    2
    >>> shifted.children[1].data  # Unchanged CountValue
    2
    """
    return postmap(node, lambda n: shift_f(n, k))


def reverse_node(knode: KNode[T]) -> KNode[T]:
    """
    Reverses the structure of the given KNode based on its type.
    Assume that SumNode has already been run-lenght-encoded

    Args:
        knode: The KNode to reverse.

    Returns:
        KNode: The reversed node.
    """
    match knode:
        case ProductNode(children):
            reversed_children = tuple(
                reverse_node(child) for child in reversed(children)
            )
            return type(knode)(reversed_children)
        case SumNode(children):
            reversed_children = frozenset(reverse_node(child) for child in children)
            return type(knode)(reversed_children)
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
            nnode = NoneValue() if isinstance(node, NoneValue) else reverse_node(node)
            return RootNode(nnode, position, colors)
        case _:
            return knode


def encode_run_length(
    primitives: Iterable[PrimitiveNode[T]],
) -> ProductNode[T] | RepeatNode[T] | PrimitiveNode[T]:
    """
    Encodes an iterable of PrimitiveNode[T] into a ProductNode[T] using run-length encoding.
    Consecutive identical PrimitiveNodes are compressed into RepeatNodes when the sequence
    length is 3 or greater.

    Args:
        primitives: An iterable of PrimitiveNode[T] objects.

    Returns:
        A ProductNode[T] containing a tuple of KNode[T] (PrimitiveNode[T] or RepeatNode[T]).
    """
    # List to store the sequence of nodes
    sequence: list[KNode[T]] = []

    # Convert the iterable to an iterator and handle the empty case
    iterator = iter(primitives)
    try:
        current = next(iterator)  # Get the first node
    except StopIteration:
        return ProductNode(tuple())  # Return an empty ProductNode if iterable is empty

    # Initialize count for the current sequence
    count = 1

    # Process each node in the iterable
    for node in iterator:
        if node == current:
            # If the node matches the current one, increment the count
            count += 1
        else:
            # If the node differs, decide how to encode the previous sequence
            if count >= 3:
                # For sequences of 3 or more, use a RepeatNode
                repeat_node = RepeatNode(current, CountValue(count))
                sequence.append(repeat_node)
            else:
                # For sequences of 1 or 2, append individual PrimitiveNodes
                sequence.extend([current] * count)
            # Update current node and reset count
            current = node
            count = 1

    # Handle the last sequence after the loop
    if count >= 3:
        repeat_node = RepeatNode(current, CountValue(count))
        sequence.append(repeat_node)
    else:
        sequence.extend([current] * count)

    # Return the sequence as a ProductNode
    if len(sequence) == 1:
        assert isinstance(sequence[0], RepeatNode | PrimitiveNode)
        return sequence[0]

    return ProductNode(tuple(sequence))


def find_repeating_pattern(
    nodes: Sequence[KNode[T]], offset: int
) -> tuple[KNode[T] | None, int, bool]:
    """
    Finds the best repeating pattern in a list of KNodes starting at the given offset, including alternating (reversed) patterns,
    optimizing for bit-length compression.

    Args:
        nodes: List of KNode[T] to search for patterns.
        offset: Starting index in the list to begin the search.

    Returns:
        Tuple of (pattern_node, count, is_reversed):
        - pattern_node: The KNode representing the repeating unit (None if no pattern found).
        - count: Number of repetitions (positive or negative based on reversal).
        - is_reversed: True if the pattern alternates with its reverse.

    """
    if offset >= len(nodes):
        return None, 0, False

    best_pattern: KNode[T] | None = None
    best_count = 0
    best_bit_gain = 0
    best_reverse = False
    max_pattern_len = (len(nodes) - offset + 1) // 2  # Need at least 2 occurrences

    for pattern_len in range(1, max_pattern_len + 1):
        pattern = nodes[offset : offset + pattern_len]
        pattern_node = iterable_to_product(pattern) if len(pattern) > 1 else pattern[0]

        for reverse in [False, True]:
            count = 1
            i = offset + pattern_len
            while i < len(nodes):
                if i + pattern_len > len(nodes):
                    break

                match = True
                segment = nodes[i : i + pattern_len]
                compare_node = (
                    reverse_node(pattern_node)
                    if (reverse and count % 2 == 1)
                    else pattern_node
                )

                # Compare the segment with the pattern or its reverse
                if len(segment) == pattern_len:
                    segment_node = (
                        iterable_to_product(segment) if len(segment) > 1 else segment[0]
                    )
                    if segment_node != compare_node:
                        match = False
                else:
                    match = False

                if match:
                    count += 1
                    i += pattern_len
                else:
                    break

            if count > 1:
                # Calculate bit-length savings
                original_bits = sum(
                    node.bit_length()
                    for node in nodes[offset : offset + pattern_len * count]
                )
                compressed = RepeatNode(
                    pattern_node, CountValue(count if not reverse else -count)
                )
                compressed_bits = compressed.bit_length()
                bit_gain = original_bits - compressed_bits

                if bit_gain > best_bit_gain:
                    best_pattern = pattern_node
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return (
        best_pattern,
        best_count if not best_reverse else -best_count,
        best_reverse,
    )


def flatten_sum(node: SumNode[T]) -> KNode[T]:
    # If there is a single node, which is not a RepeatNode, flatten the tuple
    if len(node.children) > 1:
        return node

    # Preserves iterators
    child = next(iter(node.children))
    if isinstance(child, RepeatNode):
        return node

    return child


def flatten_product(node: ProductNode[T]) -> KNode[T]:
    if len(node.children) == 0:
        raise ValueError(f"A product node contains no children: {node}")

    if len(node.children) > 1:
        return node

    return node.children[0]


def get_iterator(knodes: Collection[KNode[T]]) -> frozenset[KNode[T]]:
    """
    This function identifies if the input nodes form an arithmetic sequence and encodes it as a single RepeatNode.
    It leverages a hacky double meaning: while a standard RepeatNode(X, N) means "X for _ in range(N)",
    when used as the sole child of a SumNode, e.g., SumNode((Repeat(X, N),)), it represents
    "SumNode(tuple(shift(X, k) for k in range(N)))" if N > 0, or shifts with negative increments if N < 0.
    This compresses arithmetic enumerations cost-free, enhancing expressiveness.
    """
    nodes = frozenset(knodes)
    if len(nodes) < 2:
        return frozenset(nodes)
    if (
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
    ):
        # Guarantee a fixed order
        n_list = sorted(list(nodes), key=lambda x: str(x))
        for start_node in n_list:
            for step in [1, -1]:
                expected = {shift(start_node, step * k) for k in range(len(nodes))}
                if expected == nodes:
                    repeat_node = RepeatNode(start_node, CountValue(step * len(nodes)))
                    return frozenset([repeat_node])
    return nodes


def construct_product_node(nodes: Iterable[KNode[T]]) -> ProductNode[T]:
    """
    Constructs a ProductNode from an iterable of KNode[T], applying simplifications for efficient compression.

    - Flattens nested ProductNodes to avoid unnecessary nesting.
    - Merges adjacent PrimitiveNodes into RepeatNodes where possible using run-length encoding.
    - Combines consecutive RepeatNodes with the same base node and fixed counts.
    - Preserves SumNodes as-is to maintain logical structure.

    Args:
        nodes: An iterable of KNode[T] to combine into a ProductNode.

    Returns:
        A ProductNode[T] representing the simplified sequence.
    """
    # Convert the iterable to a list for manipulation
    simplified: list[KNode[T]] = []
    for node in nodes:
        match node:
            case ProductNode(children):
                simplified.extend(children)  # Flatten nested ProductNodes
            case _:
                simplified.append(node)  # Keep other nodes as-is

    # Simplify the flattened list
    i = 0
    while i < len(simplified):
        current = simplified[i]
        match current:
            case PrimitiveNode():
                # Collect consecutive PrimitiveNodes
                primitives = [current]
                j = i + 1
                while j < len(simplified) and isinstance(simplified[j], PrimitiveNode):
                    primitives.append(simplified[j])  # type: ignore[reportArgumentType]
                    j += 1
                if len(primitives) > 1:
                    # Replace with run-length encoded version (assumes encode_run_length exists)
                    encoded = encode_run_length(primitives)
                    if isinstance(encoded, ProductNode):
                        simplified[i:j] = list(
                            encoded.children
                        )  # Insert the children directly                    i += 1
                        i += len(encoded.children)  # Move index past inserted elements
                    else:
                        simplified[i:j] = [encoded]
                        i += 1
                else:
                    i += 1
            case RepeatNode(node=base, count=CountValue(count)) if i + 1 < len(
                simplified
            ):
                next_node = simplified[i + 1]
                match next_node:
                    case RepeatNode(node=next_base, count=CountValue(next_count)) if (
                        base == next_base
                    ):
                        # Combine consecutive RepeatNodes with the same base
                        combined_count = CountValue(count + next_count)
                        simplified[i] = RepeatNode(base, combined_count)
                        del simplified[i + 1]
                    case _:
                        i += 1
            case _:
                i += 1  # Move past non-mergeable nodes (e.g., SumNode)

    return ProductNode(tuple(simplified))


def iterable_to_product(iterable: Iterable[KNode[T]]) -> KNode[T]:
    nodes: list[KNode[T]] = list(iterable)
    if not nodes:
        raise ValueError("Empty iterable. Can't produce a product out of it")
    elif len(nodes) == 1:
        return nodes[0]
    else:
        nnodes = construct_product_node(nodes)
        return flatten_product(nnodes)


def iterable_to_sum(
    iterable: Iterable[KNode[T]], compress_iterator=False
) -> KNode[T] | None:
    nodes: frozenset[KNode[T]] = frozenset(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return next(iter(nodes))
    elif compress_iterator:
        return SumNode(get_iterator(nodes))
    else:
        return SumNode(nodes)


def factorize_tuple(node: KNode[T]) -> KNode[T]:
    """
    Compresses a ProductNode or SumNode by detecting and encoding repeating patterns with RepeatNodes.

    Args:
        node: A KNode[T], typically a ProductNode or SumNode, to factorize.

    Returns:
        A new KNode[T] with repeating patterns compressed.
    """
    if isinstance(node, SumNode):
        # For SumNode, delegate to get_iterator for arithmetic sequence compression
        children = get_iterator(node.children)
        # If there is a no repeat single node, flatten the tuple
        # For the iterator hack, no unpacking
        return flatten_sum(SumNode(children))

    if not isinstance(node, ProductNode):
        return node

    # Handle ProductNode compression
    children = list(node.children)
    if len(children) < 2:
        return node

    simplified: list[KNode[T]] = []
    i = 0
    while i < len(children):
        pattern, count, is_reversed = find_repeating_pattern(children, i)
        if pattern is not None and abs(count) > 1:
            # Calculate bit lengths
            repeat_count = abs(count)
            original_bits = sum(
                child.bit_length()
                for child in children[
                    i : i
                    + repeat_count
                    * (len(pattern.children) if isinstance(pattern, ProductNode) else 1)
                ]
            )
            repeat_node = RepeatNode(pattern, CountValue(count))
            compressed_bits = repeat_node.bit_length()

            # Only use RepeatNode if it reduces bit length
            if compressed_bits < original_bits:
                simplified.append(repeat_node)
                i += repeat_count * (
                    len(pattern.children) if isinstance(pattern, ProductNode) else 1
                )
            else:
                # If no bit savings, append the original nodes one by one
                num_nodes = (
                    len(pattern.children) if isinstance(pattern, ProductNode) else 1
                )
                simplified.extend(children[i : i + num_nodes])
                i += num_nodes

        else:
            simplified.append(children[i])
            i += 1

    # Reconstruct the ProductNode with simplified children
    if len(simplified) == 1 and simplified[0] == node:
        return node  # Avoid unnecessary reconstruction
    return cast(KNode, iterable_to_product(simplified))


def expand_repeats(
    node: KNode[T],
) -> KNode[T]:
    """Used to uncompress knodes"""

    def expand_repeat_f(knode: KNode) -> KNode:
        """Expand repeats"""
        if isinstance(knode, RepeatNode):
            if isinstance(knode.count, VariableNode):
                raise TypeError("Trying to uncompress a variable repeat")
            return cast(KNode, ProductNode((knode.node,) * knode.count.value))
        else:
            return knode

    expanded = postmap(node, expand_repeat_f, factorize=False)
    return cast(SumNode[T] | ProductNode[T] | PrimitiveNode[T], expanded)


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
