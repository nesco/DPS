"""
Kolmogorov Tree: An abstract-syntax tree structure forming to represent non-deterministic bit-length-aware programs.
TL;DR: An bitlengthaware AST for bitlengthaware rosetrees

This module implements a tree structure for representing non-deterministic programs,
for example to describe shapes using a non-deterministic program representing 2D grid movements (see syntax_tree.py). The representation
aims to approximate Kolmogorov complexity through minimum description length.

The tree structure consists of:
- ProductNodes for deterministic sequences
- SumNodes for non-deterministic branching
- RepeatNodes for repetition extraction for ProductNodes, an artifact of DFS
- NestedNodes is like RepeatNodes but for SumNodes, an artificat of BFS
- SymbolNodes for pattern abstraction and reuse
- Variable binding for lambda abstraction

Key Features:
- Computable bit-length metrics for complexity approximation
- Pattern extraction and memorization via symbol table
- Support for 8-directional grid movements
- "lambda abstraction" through variable binding flattened into functions with arity
- Nested pattern detection and reuse

The tree can be used to:
1. Represent shapes as non-deterministic programs
2. Extract common patterns across multiple programs
3. Measure and optimize program complexity
4. Enable search over program space guided by complexity


In a way, BitLengthAware is a type encapsulation, so the types "know" their bit length, like types that knows
their gradients in deep learning.

Example Usage:
```python
# Create a simple pattern
pattern = create_moves_sequence("2323")  # Right-Down-Right-Down
# Repeat it 3 times
repeated = RepeatNode(pattern, 3)
# Add non-deterministic branching
program = SumNode([repeated, create_rect(3, 3)])
# Create full tree with starting position and colors
tree = KolmogorovTree(RootNode((0,0), {1}, program))
```

Non-trial combinations:
- SumNode({RepeatNode()}) <- single element, which is a repeat, that get transformed into an iterator, otherwise for single elements SumNodes might need to get flattened
- RepeatNode(SumNode, count) <- should be nested alternatives
- RootNode should either be top level, or in a SumNode containing only RootNodes
"""

# TO-DO:
# Usually when constructed through a graph DFS, sum nodes always ends-up at the end of a product node if in a sequence
# It could lead to further compression, if the end of the sum node repeats
# E.G 111 [134,244,334] -> 111[13, 24, 33]4

import functools
import copy

from collections import Counter, defaultdict, deque
from collections.abc import Collection
from typing import (
    Callable,
    Iterable,
    Sequence,
    cast,
)

from localtypes import (
    BitLengthAware,
    Coord,  # noqa: F401 - re-exported for backwards compatibility
    Primitive,
)
from utils.tree_functionals import (
    postorder_map,
    preorder_map,
)

# Import primitives from the new module (re-exported for backwards compatibility)
from kolmogorov_tree.primitives import (  # noqa: F401
    ARCBitLength,
    Alphabet,
    BitLength,
    CoordValue,
    CountValue,
    IndexValue,
    MoveValue,
    NoneValue,
    PaletteValue,
    T,
    VariableValue,
)

# Import nodes from the new module (re-exported for backwards compatibility)
from kolmogorov_tree.nodes import (  # noqa: F401
    CollectionNode,
    KNode,
    NestedNode,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    SymbolNode,
    Uncompressed,
    Unsymbolized,
    VariableNode,
)

# Import traversal utilities
from kolmogorov_tree.traversal import (  # noqa: F401
    breadth_first_preorder_knode,
    children,
    depth,
    depth_first_preorder_bitlengthaware,
    get_subvalues,
    next_layer,
)

# Import factory functions
from kolmogorov_tree.factories import (  # noqa: F401
    create_move_node,
    create_moves_sequence,
    create_rect,
    create_variable_node,
    cv,
)

# Import parsing utilities
from kolmogorov_tree.parsing import (  # noqa: F401
    split_top_level_arguments,
    str_to_knode,
    str_to_repr,
)

# Import predicate/inspection utilities
from kolmogorov_tree.predicates import (  # noqa: F401
    arity,
    contained_symbols,
    is_abstraction,
    is_symbolized,
)

## Functions on KNodes


# Helpers
def shift_f(node: KNode[T], k: int) -> KNode[T]:
    if isinstance(node, PrimitiveNode) and isinstance(node.value, Alphabet):
        shifted_value = node.value.shift(k)
        return PrimitiveNode[T](shifted_value)
    return node


type Parameters = tuple[BitLengthAware, ...]


def extract_template(knode: KNode[T]) -> list[tuple[KNode[T], Parameters]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions: list[tuple[KNode[T], Parameters]] = []

    # If the node is already a lambda-abstraction
    # It should not be abstracted further
    if is_abstraction(knode):
        return abstractions

    match knode:
        case ProductNode(children) if len(children) > 2:
            # Abstract up to two distinct elements for now
            child_counter = Counter(children)  # Count occurrences of each child
            child_set = list(child_counter.keys())
            max_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[
                :2
            ]  # I don't know why it works better when it's 4 here instead of 2

            # Abstract the most frequent/largest child
            nodes1 = tuple(
                VariableNode(VariableValue(0)) if c == max_children[0] else c
                for c in children
            )
            abstractions.append((ProductNode(nodes1), (max_children[0],)))

            # If there are at least two distinct children and length > 2
            if len(max_children) > 1 and len(children) > 2:
                # Abstract the second most frequent/largest child
                nodes2 = tuple(
                    VariableNode(VariableValue(0))
                    if c == max_children[1]
                    else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes2), (max_children[0],)))

                # Then absttract the top two
                nodes3 = tuple(
                    VariableNode(VariableValue(max_children.index(c)))
                    if c in max_children
                    else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes3), tuple(max_children)))
        case SumNode(children) if len(children) > 2:
            # Abstract up to two distinct elements for now
            child_counter = Counter(children)  # Count occurrences of each child
            child_set = list(child_counter.keys())
            max_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]

            # Abstract the most frequent/largest child
            nodes1 = frozenset(
                VariableNode(VariableValue(0)) if c == max_children[0] else c
                for c in children
            )
            abstractions.append((SumNode(nodes1), (max_children[0],)))

            # If there are at least two distinct children and length > 2
            if len(max_children) > 1 and len(children) > 2:
                # Abstract the second most frequent/largest child
                nodes2 = frozenset(
                    VariableNode(VariableValue(0))
                    if c == max_children[1]
                    else c
                    for c in children
                )
                abstractions.append((SumNode(nodes2), (max_children[0],)))

                # Then absttract the top two
                nodes3 = frozenset(
                    VariableNode(VariableValue(max_children.index(c)))
                    if c in max_children
                    else c
                    for c in children
                )
                abstractions.append((SumNode(nodes3), tuple(max_children)))
        case RepeatNode(node, count):
            # For a RepeatNode, either the node or the count can be abstracted
            abstractions.extend(
                [
                    (
                        RepeatNode(VariableNode(VariableValue(0)), count),
                        (node,),
                    ),
                    (
                        RepeatNode(node, VariableNode(VariableValue(0))),
                        (count,),
                    ),
                ]
            )
        case NestedNode(index, node, count):
            abstractions.extend(
                [
                    (
                        NestedNode(
                            index, VariableNode(VariableValue(0)), count
                        ),
                        (node,),
                    ),
                    (
                        NestedNode(
                            index,
                            node,
                            VariableNode(VariableValue(0)),
                        ),
                        (count,),
                    ),
                    (
                        NestedNode(
                            index,
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(1)),
                        ),
                        (
                            node,
                            count,
                        ),
                    ),
                ]
            )
        case RootNode(node, position, colors):
            # In the case of a RootNode,
            # because of ARC, we don't want the position to be memorized alone
            # It would defeat the objectification lattice step'
            # TODO
            # See if the above is still applicable
            # Understand the further 'colors.value' == 1 condition
            abstractions.extend(
                [
                    # 1 parameter
                    (
                        RootNode(
                            VariableNode(VariableValue(0)), position, colors
                        ),
                        (node,),
                    ),
                    (
                        RootNode(node, VariableNode(VariableValue(0)), colors),
                        (position,),
                    ),
                    # 2 parameters
                    (
                        RootNode(
                            VariableNode(VariableValue(0)),
                            position,
                            VariableNode(VariableValue(1)),
                        ),
                        (node, colors),
                    ),
                ]
            )
            if len(colors.value) == 1:  # type: ignore -> because of 'is_abstraction', it can't be a Variable'
                abstractions.append(
                    (
                        RootNode(
                            node,
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(1)),
                        ),
                        (position, colors),
                    )
                )
        case RectNode(heigth, width):
            if heigth == width:
                abstractions.append(
                    (
                        RectNode(
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(0)),
                        ),
                        (heigth,),
                    )
                )
                abstractions.extend(
                    [
                        (
                            RectNode(VariableNode(VariableValue(0)), width),
                            (heigth,),
                        ),
                        (
                            RectNode(heigth, VariableNode(VariableValue(0))),
                            (width,),
                        ),
                    ]
                )
        case _:
            pass

    return abstractions


def extract_nested_sum_template(
    snode: SumNode[T],
) -> tuple[SumNode[T], SumNode[T]] | None:
    """
    Like generate abstractions but for structures like [0[]|4]

    Returns
        tuple[SumNode[T], SumNode[T]]: (template, parameter)
    """
    max_product = max(
        [node for node in snode.children if isinstance(node, ProductNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_product:
        return None

    max_sum = max(
        [node for node in max_product.children if isinstance(node, SumNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_sum:
        return None

    new_product = ProductNode(
        tuple(
            node if node != max_sum else VariableNode(VariableValue(0))
            for node in max_product.children
        )
    )

    new_sum = SumNode(
        frozenset(
            {
                node if node != max_product else new_product
                for node in snode.children
            }
        )
    )

    return new_sum, max_sum


def extract_nested_product_template(
    pnode: ProductNode[T],
) -> tuple[ProductNode[T], ProductNode[T]] | None:
    """
    Like generate abstractions but for structures like [0[]|4]

    Returns
        tuple[SumNode[T], SumNode[T]]: (template, parameter)
    """
    max_sum = max(
        [node for node in pnode.children if isinstance(node, SumNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_sum:
        return None

    max_product = max(
        [node for node in max_sum.children if isinstance(node, ProductNode)],
        key=lambda x: x.bit_length(),
        default=None,
    )

    if not max_product:
        return None

    new_sum = SumNode(
        frozenset(
            {
                node if node != max_product else VariableNode(VariableValue(0))
                for node in max_sum.children
            }
        )
    )

    new_product = ProductNode(
        tuple(node if node != max_sum else new_sum for node in pnode.children)
    )

    return new_product, max_product


def detect_recursive_collection(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[KNode[T], KNode[T], int] | None:
    """
    Detects a recursive pattern in a SumNode, returning the common template, terminal node, and recursion count.

    Args:
        snode: The SumNode to analyze.

    Returns:
        tuple[KNode[T], KNode[T], int]: (template, terminal_node, count) if a recursive pattern is found, else None.
    """
    count = 0
    current = knode
    common_template = None

    while True:
        assert isinstance(current, type(knode))
        extraction = (
            extract_nested_sum_template(current)
            if isinstance(current, SumNode)
            else extract_nested_product_template(current)
        )

        if extraction is None:
            break
        template, parameter = extraction

        if count == 0:
            common_template = (
                template  # Set the template on the first iteration
            )
        elif template != common_template:
            break  # Stop if the template changes

        if not isinstance(parameter, type(knode)):
            break  # Stop if the parameter isn’t a SumNode

        count += 1
        current = parameter

    if count >= 1 and common_template is not None:
        return common_template, current, count
    return None


def nested_collection_to_nested_node(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[NestedNode[T], KNode[T]] | None:
    """
    Transforms a CollectionNode with a recursive pattern into a NestedNode, returning the node and its template.

    Args:
        snode: The SumNode to transform.

    Returns:
        tuple[NestedNode[T], KNode[T]]: (nested_node, template) if a recursive pattern is found, else None.
    """
    recursive_info = detect_recursive_collection(knode)
    if recursive_info is None:
        return None

    template, terminal_node, count = recursive_info
    # Use a placeholder index; the symbol table must be updated externally
    nested_node = NestedNode(
        index=IndexValue(0), node=terminal_node, count=CountValue(count)
    )
    return nested_node, template


def get_iterator_old(knodes: Collection[KNode[T]]) -> frozenset[KNode[T]]:
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
    if all(
        isinstance(n, PrimitiveNode) and isinstance(n.value, MoveValue)
        for n in nodes
    ) or all(isinstance(n, RepeatNode) for n in nodes):
        for start_node in nodes:
            for step in [1, -1]:
                expected = {
                    shift(start_node, step * k) for k in range(len(nodes))
                }
                if expected == nodes:
                    repeat_node = RepeatNode(
                        start_node, CountValue(step * len(nodes))
                    )
                    return frozenset([repeat_node])
    return nodes


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
                expected = {
                    shift(start_node, step * k) for k in range(len(nodes))
                }
                if expected == nodes:
                    repeat_node = RepeatNode(
                        start_node, CountValue(step * len(nodes))
                    )
                    return frozenset([repeat_node])
    return nodes


# Only useful for ARC? What if the alphabet is too large?
def get_iterator2(nodes: Iterable[KNode[T]]) -> tuple[KNode[T], ...]:
    """
    This function identifies if the input nodes form an arithmetic sequence and encodes it as a single RepeatNode.
    It leverages a hacky double meaning: while a standard RepeatNode(X, N) means "X for _ in range(N)",
    when used as the sole child of a SumNode, e.g., SumNode((Repeat(X, N),)), it represents
    "SumNode(tuple(shift(X, k) for k in range(N)))" if N > 0, or shifts with negative increments if N < 0.
    This compresses arithmetic enumerations cost-free, enhancing expressiveness.
    """
    node_ls: list[KNode[T]] = list(nodes)
    if len(node_ls) < 2:
        return tuple(node_ls)

    # Check if the sequence has a consistent increment (1 or -1)
    prev = node_ls[0]
    curr = node_ls[1]

    # Determine the increment based on the first pair
    if curr == shift(prev, 1):
        increment = 1
    elif curr == shift(prev, -1):
        increment = -1
    else:
        return tuple(node_ls)

    # Validate the entire sequence
    for i in range(2, len(node_ls)):
        prev = node_ls[i - 1]
        curr = node_ls[i]
        if curr != shift(prev, increment):
            return tuple(node_ls)

    # If we reach here, the sequence is arithmetic; encode it as a RepeatNode
    # Use increment * length to encode direction and length
    return (RepeatNode(node_ls[0], CountValue(increment * len(node_ls))),)


# Constructors


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
                while j < len(simplified) and isinstance(
                    simplified[j], PrimitiveNode
                ):
                    primitives.append(simplified[j])  # type: ignore[reportArgumentType]
                    j += 1
                if len(primitives) > 1:
                    # Replace with run-length encoded version (assumes encode_run_length exists)
                    encoded = encode_run_length(primitives)
                    if isinstance(encoded, ProductNode):
                        simplified[i:j] = list(
                            encoded.children
                        )  # Insert the children directly                    i += 1
                        i += len(
                            encoded.children
                        )  # Move index past inserted elements
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
                    case RepeatNode(
                        node=next_base, count=CountValue(next_count)
                    ) if base == next_base:
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


# Parsing functions are now in kolmogorov_tree.parsing (imported above)

# Basic operations on Nodes
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
            reversed_children = frozenset(
                reverse_node(child) for child in children
            )
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
            nnode = (
                NoneValue()
                if isinstance(node, NoneValue)
                else reverse_node(node)
            )
            return RootNode(nnode, position, colors)
        case _:
            return knode


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
        return ProductNode(
            tuple()
        )  # Return an empty ProductNode if iterable is empty

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


# Predicate/inspection functions are now in kolmogorov_tree.predicates (imported above)

# Compression
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
    max_pattern_len = (
        len(nodes) - offset + 1
    ) // 2  # Need at least 2 occurrences

    for pattern_len in range(1, max_pattern_len + 1):
        pattern = nodes[offset : offset + pattern_len]
        pattern_node = (
            iterable_to_product(pattern) if len(pattern) > 1 else pattern[0]
        )

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
                        iterable_to_product(segment)
                        if len(segment) > 1
                        else segment[0]
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
        first = next(iter(node.children))
        if (
            isinstance(first, RepeatNode)
            and isinstance(first.count, CountValue)
            and first.count == 5
        ):
            print("TEST")
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
                    * (
                        len(pattern.children)
                        if isinstance(pattern, ProductNode)
                        else 1
                    )
                ]
            )
            repeat_node = RepeatNode(pattern, CountValue(count))
            compressed_bits = repeat_node.bit_length()

            # Only use RepeatNode if it reduces bit length
            if compressed_bits < original_bits:
                simplified.append(repeat_node)
                i += repeat_count * (
                    len(pattern.children)
                    if isinstance(pattern, ProductNode)
                    else 1
                )
            else:
                # If no bit savings, append the original nodes one by one
                num_nodes = (
                    len(pattern.children)
                    if isinstance(pattern, ProductNode)
                    else 1
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


# High-order functions


def postmap(
    knode: KNode[T], f: Callable[[KNode[T]], KNode[T]], factorize: bool = True
) -> KNode[T]:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.
        factorize: Wether to automatically factorize the TupleNodes of the tree

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
        factorize: Wether to automatically factorize the TupleNodes of the tree

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    return preorder_map(
        knode,
        f,
        children,
        lambda node, kids: reconstruct_knode(node, kids, factorize),
    )


# Decompression
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


# Symbol resolution and pattern finding

Bindings = dict[int, BitLengthAware]


def matches(pattern: KNode[T], subtree: KNode[T]) -> Bindings | None:
    """
    Determines if a pattern matches a subtree, returning variable bindings if successful.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree to match against.

    Returns:
        Bindings | None: A dictionary of variable bindings if the match succeeds, or None if it fails.
    """
    if not isinstance(pattern, KNode) or not isinstance(subtree, KNode):
        raise TypeError("Both pattern and subtree must be KNode instances.")
    bindings: Bindings = {}
    if unify(pattern, subtree, bindings):
        return bindings
    return None


def unify_sum_children(
    p_idx: int,  # Index of the current pattern child being matched
    pattern_list: list[KNode[T]],  # Sorted list of pattern children
    subtree_list: list[KNode[T]],  # Sorted list of subtree children
    subtree_used: list[
        bool
    ],  # Tracks which subtree children are already matched
    bindings: Bindings,
) -> Bindings | None:
    """
    Helper function using backtracking on sorted lists to find a deterministic
    unification mapping between SumNode children.

    Args:
        p_idx: Current index in pattern_list being processed.
        pattern_list: Sorted list of pattern children.
        subtree_list: Sorted list of subtree children.
        subtree_used: Boolean mask indicating used subtree children.
        bindings: Current accumulated bindings for this search path.

    Returns:
        Updated bindings if a valid matching is found, otherwise None.
    """
    # Base case: All pattern children have been successfully matched.
    if p_idx == len(pattern_list):
        return bindings

    p_child = pattern_list[p_idx]

    # Iterate through potential matches in the sorted subtree list
    for s_idx in range(len(subtree_list)):
        # Check if this subtree child is already used in the current mapping
        if not subtree_used[s_idx]:
            s_child = subtree_list[s_idx]

            # Create a copy of bindings to explore this specific pair's unification
            current_bindings = bindings.copy()

            # Attempt to unify the chosen pair (p_child, s_child)
            if unify(p_child, s_child, current_bindings):
                # If the pair unifies successfully:
                # 1. Mark the subtree child as used for this path
                subtree_used[s_idx] = True

                # 2. Recursively try to match the *next* pattern child (p_idx + 1)
                result_bindings = unify_sum_children(
                    p_idx + 1,
                    pattern_list,
                    subtree_list,
                    subtree_used,
                    current_bindings,  # Pass the updated bindings
                )

                # 3. Check if the recursive call was successful
                if result_bindings is not None:
                    # Success! A complete valid matching was found. Return it.
                    return result_bindings

                # 4. Backtrack: If the recursive call failed, unmark the
                #    subtree child so it can be tried with other pattern children
                #    in alternative branches of the search.
                subtree_used[s_idx] = False
                # Continue the loop to try matching p_child with the next unused s_child

    # If the loop completes without returning, it means p_child could not be
    # successfully matched with any *available* s_child in a way that allowed the
    # rest of the pattern children to be matched. Backtrack by returning None.
    return None


def unify(
    pattern: BitLengthAware, subtree: BitLengthAware, bindings: Bindings
) -> bool:
    """
    Recursively unifies a pattern with a subtree, updating bindings in place.

    Args:
        pattern: The pattern to match, which may contain variables.
        subtree: The concrete subtree or primitive to match against.
        bindings: Current bindings of variables to subtrees or primitives, modified in place.

    Returns:
        bool: True if unification succeeds, False if it fails.
    """
    if isinstance(pattern, VariableNode):
        index = pattern.index.value
        if index in bindings:
            return bindings[index] == subtree
        bindings[index] = subtree
        return True

    if not isinstance(subtree, type(pattern)):
        return False

    match pattern:
        case Primitive(value):
            assert isinstance(subtree, Primitive)
            return value == subtree.value
            # Inside the unify function...
        case SumNode(children):
            assert isinstance(subtree, SumNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            if not children:
                return True

            # 1. Create sorted lists based on a canonical key
            # Sort using the  string representation
            sorted_pattern_children = sorted(children, key=str)
            sorted_subtree_children = sorted(s_children, key=str)
            # 2. Initialize the tracking list for used subtree children
            subtree_used = [False] * len(sorted_subtree_children)

            # 3. Call the deterministic backtracking helper
            initial_bindings = bindings.copy()
            successful_bindings = unify_sum_children(
                0,  # Start matching the first pattern child (index 0)
                sorted_pattern_children,
                sorted_subtree_children,
                subtree_used,
                initial_bindings,
            )

            if successful_bindings is not None:
                # Update original bindings *in place*
                bindings.clear()
                bindings.update(successful_bindings)
                return True
            else:
                return False

        case ProductNode(children):
            assert isinstance(subtree, ProductNode)
            s_children = subtree.children
            if len(children) != len(s_children):
                return False
            for p_child, s_child in zip(children, s_children):
                if not unify(p_child, s_child, bindings):
                    return False
            return True
        case RepeatNode(node, count):
            subtree = cast(RepeatNode, subtree)
            s_node = subtree.node
            s_count = subtree.count
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != s_count:
                        return False
                else:
                    bindings[index] = s_count
            elif count != s_count:
                return False
            return unify(node, s_node, bindings)
        case NestedNode(index, node, count):
            subtree = cast(NestedNode, subtree)
            s_node = subtree.node
            s_count = subtree.count
            if isinstance(count, VariableNode):
                index = count.index.value
                if index in bindings:
                    if bindings[index] != s_count:
                        return False
                else:
                    bindings[index] = s_count
            elif count != s_count:
                return False
            return unify(node, s_node, bindings)

        case RootNode(node, position, colors):
            subtree = cast(RootNode, subtree)
            if not unify(position, subtree.position, bindings):
                return False
            if not unify(colors, subtree.colors, bindings):
                return False
            return unify(node, subtree.node, bindings)
        case SymbolNode(index, parameters):
            subtree = cast(SymbolNode, subtree)
            if index != subtree.index or len(parameters) != len(
                subtree.parameters
            ):
                return False
            for param, s_param in zip(parameters, subtree.parameters):
                if not unify(param, s_param, bindings):
                    return False
            return True
        case RectNode(height, width):
            subtree = cast(RectNode, subtree)
            if not unify(height, subtree.height, bindings):
                return False
            if not unify(width, subtree.width, bindings):
                return False
            return True
        case _:
            return pattern == subtree


def abstract_node(
    index: IndexValue, pattern: KNode[T], node: KNode[T]
) -> KNode[T]:
    """
    If node matches the pattern, it replaces it by a SymbolNode with the right bindings.
    Else, it returns the node as is.
    """
    if pattern == node:
        return SymbolNode(index, ())

    bindings = matches(pattern, node)

    if bindings is not None:
        params = tuple(
            binding
            for j in range(max(bindings.keys(), default=-1) + 1)
            if (binding := bindings.get(j, None)) is not None
        )
        return SymbolNode(index, params)

    return node


def node_to_symbolized_node(
    index: IndexValue, pattern: KNode[T], knode: KNode[T]
) -> KNode[T]:
    return postmap(knode, lambda node: abstract_node(index, pattern, node))


def substitute_variables_deprecated(
    abstraction: KNode[T], params: Parameters
) -> KNode[T]:
    """
    Helper function to substitute variables, including those that are not traversed
    during a simple node traversal
    """
    print(f"abstraction: {abstraction}, params: {params}")
    match abstraction:
        case VariableNode(index) if index.value < len(params):
            node = params[index.value]
            if not isinstance(node, KNode):
                raise TypeError(
                    f"Trying to substitute a non-node parameter to a variable encountered during node traversal: {[node]}"
                )
            return node
        case RepeatNode(node, VariableNode(index)) if index.value < len(params):
            count = params[index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return RepeatNode(node, count)
        case NestedNode(index, node, count) if isinstance(
            count, VariableNode
        ) and count.index.value < len(params):
            count = params[count.index.value]
            if not isinstance(count, CountValue):
                raise TypeError(
                    f"Trying to substitute a count variable to a wrong parameter: {count}"
                )
            return NestedNode(index, node, count)
        case RootNode(node, VariableNode(index1), VariableNode(index2)) if (
            index1.value < len(params) and index2.value < len(params)
        ):
            position = params[index1.value]
            colors = params[index2.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, VariableNode(index), colors) if index.value < len(
            params
        ):
            position = params[index.value]
            if not isinstance(position, CoordValue):
                raise TypeError(
                    f"Trying to substitute a position variable to a wrong parameter: {position}"
                )
            return RootNode(node, position, colors)
        case RootNode(node, position, VariableNode(index)) if index.value < len(
            params
        ):
            colors = params[index.value]
            if not isinstance(colors, PaletteValue):
                raise TypeError(
                    f"Trying to substitute a colors variable to a wrong parameter: {colors}"
                )
            return RootNode(node, position, colors)
        case RectNode(VariableNode(index1), VariableNode(index2)) if (
            index1.value < len(params) and index2.value < len(params)
        ):
            height = params[index1.value]
            width = params[index2.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case RectNode(VariableNode(index), width) if index.value < len(params):
            height = params[index.value]
            if not isinstance(height, CountValue):
                raise TypeError(
                    f"Trying to substitute a height variable to a wrong parameter: {height}"
                )
            return RectNode(height, width)
        case RectNode(height, VariableNode(index)) if index.value < len(params):
            width = params[index.value]
            if not isinstance(width, CountValue):
                raise TypeError(
                    f"Trying to substitute a width variable to a wrong parameter: {width}"
                )
            return RectNode(height, width)
        case _:
            return abstraction


def variable_to_param(node: VariableNode, params: Parameters) -> BitLengthAware:
    if node.index.value < len(params):
        value = params[node.index.value]
        return value
    return node
    # raise ValueError(f"Unknown variable {node.index.value} for {params}")


def substitute_variables(knode: KNode[T], params: Parameters) -> KNode[T]:
    # Avoid infinite recursion on VariableNodes by treating them immediately
    # And returning directly
    if isinstance(knode, VariableNode):
        node = variable_to_param(knode, params)
        if not isinstance(node, KNode):
            raise TypeError(f"{node} is not a KNode")
        return node

    # Iterating over fields
    nfields = {}
    for field_name, value in vars(knode).items():
        match value:
            # Replace directly variables to avoid hitting the TypeError
            case VariableNode():
                nvalue = variable_to_param(value, params)
            case frozenset():
                nvalue = frozenset(
                    {
                        substitute_variables(el, params)
                        if isinstance(el, KNode)
                        else el
                        for el in value
                    }
                )
            case tuple():
                nvalue = tuple(
                    substitute_variables(el, params)
                    if isinstance(el, KNode)
                    else el
                    for el in value
                )
            case KNode():
                nvalue = substitute_variables(value, params)
            case _:
                nvalue = value

        nfields[field_name] = nvalue

    return type(knode)(**nfields)


@functools.cache
def reduce_abstraction(abstraction: KNode[T], params: Parameters) -> KNode[T]:
    """
    Substitutes variable placeholders in the template with the corresponding parameters.

    This function recursively traverses the template node and replaces VariableNodes
    with the provided parameters. It also handles composite nodes by substituting variables
    in their children or attributes.

    Args:
        abstraction (KNode): The abstract node to process.
        params (tuple[Any, ...]): The parameters to substitute for variables.

    Returns:
        KNode: The abstraction with variables replaced by parameters.
    """
    # return postmap(abstraction, lambda node: substitute_variables(node, params))
    return substitute_variables(abstraction, params)


@functools.cache
def resolve_symbols(knode: KNode[T], symbols: Sequence[KNode[T]]) -> KNode[T]:
    """
    Resolves symbolic references recursively using the symbol table.

    This function traverses the Kolmogorov Tree and replaces SymbolNodes with their
    definitions from the symbol table, handling parameter substitution. It recursively
    resolves all sub-nodes for composite node types.

    Args:
        node (KNode): The node to resolve.
        symbols (list[KNode]): The list of symbol definitions.

    Returns:
        KNode: The resolved node with symbols expanded.
    """

    def resolve_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and 0 <= node.index.value < len(
            symbols
        ):
            return reduce_abstraction(
                symbols[node.index.value], node.parameters
            )
        return node

    # A symbolic node can be resolved into another symbolic node
    def resolve_until(node: KNode[T]) -> KNode[T]:
        while isinstance(node, SymbolNode) and node.index.value < len(symbols):
            node = resolve_f(node)
        return node

    node = premap(knode, resolve_until, factorize=False)

    return node


def extract_nested_patterns(
    symbol_table: list[KNode[T]],
    tree: KNode[T],
) -> KNode[T]:
    """
    Traverses a Kolmogorov Tree to identify and replace nested patterns with NestedNodes,
    extracting their templates into a symbol table.

    Args:
        tree: The input Kolmogorov Tree (KNode[T]).

    Returns:
        KNode[T]: The transformed tree with NestedNodes replacing nested patterns.

    Edge-Effects:
        Fill the given symbol table with templates extracted from the nested patterns.
    """

    def mapping_function(node: KNode[T]) -> KNode[T]:
        """
        Mapping function for postmap to detect nested patterns and replace them with NestedNodes.

        Args:
            node: The current node being processed.

        Returns:
            KNode[T]: The transformed node (either a NestedNode or the original node).
        """
        # Only process SumNode or ProductNode for nested patterns
        if isinstance(node, (SumNode, ProductNode)):
            result = nested_collection_to_nested_node(node)
            if result is not None:
                nested_node, template = result
                # Greedy test, it might be better because the template might be shared with others
                if (
                    nested_node.bit_length() + template.bit_length()
                    <= node.bit_length()
                ):
                    # Add the template to the symbol table and get its index
                    if template in symbol_table:
                        index = symbol_table.index(template)
                    else:
                        index = len(symbol_table)
                        symbol_table.append(template)
                    # Create a new NestedNode with the correct index
                    # Since NestedNode is frozen, instantiate a new one
                    new_nested_node = NestedNode(
                        IndexValue(index), nested_node.node, nested_node.count
                    )
                    return new_nested_node
        # Return the node unchanged if no pattern is detected
        return node

    # Apply postmap to transform the tree
    new_tree = premap(tree, mapping_function)
    return new_tree


def expand_nested_node(
    nested_node: NestedNode[T], symbol_table: Sequence[KNode[T]]
) -> KNode[T]:
    """Expands a NestedNode using its template from the symbol table."""
    template = symbol_table[nested_node.index.value]
    current = nested_node.node

    if not isinstance(nested_node.count, CountValue):
        raise ValueError(
            f"Trying to expand a nested node with a variable count: {nested_node}"
        )

    for _ in range(nested_node.count.value):
        current = postmap(
            template, lambda node: substitute_variables(node, (current,))
        )
    return current


def expand_all_nested_nodes(knode: KNode[T], symbol_table: Sequence[KNode[T]]):
    """
    Expand all the inner NestedNodes of a given node.

    Args:
        knode: KNode possibly containing nested nodes
        symbol_table: Symbol Table containing all the template referenced by NestedNode. It will fail otherwise.
    Returns:
        KNode[T]: node free of NestedNodes
    """

    def expand_f(node: KNode[T]) -> KNode[T]:
        if isinstance(node, NestedNode):
            return expand_nested_node(node, symbol_table)
        return node

    return postmap(knode, expand_f)


def find_symbol_candidates(
    trees: Sequence[KNode[T]],
    min_occurrences: int = 2,
    max_patterns: int = 10,
) -> list[KNode]:
    """
    Identifies frequent concrete and abstracted subtrees across multiple KolmogorovTrees.
    Returns the top patterns ranked by bit-length savings.
    """
    # Step 1: Collect all subtrees
    all_subtrees = []
    for tree in trees:
        all_subtrees.extend(breadth_first_preorder_knode(tree))

    # Step 2: Count frequencies and track matches
    pattern_counter = Counter()
    pattern_matches = defaultdict(
        list
    )  # Maps pattern to set of matched subtrees

    for subtree in all_subtrees:
        pattern_counter[subtree] += 1
        pattern_matches[subtree].append(subtree)
        for abs_pattern, params in extract_template(subtree):
            pattern_counter[abs_pattern] += 1
            pattern_matches[abs_pattern].append(subtree)

    # Step 3: Filter patterns
    common_patterns = []
    seen_patterns = set()

    for pattern, count in pattern_counter.items():
        if count < min_occurrences or pattern in seen_patterns:
            continue
        if any(
            isinstance(n, VariableNode)
            for n in breadth_first_preorder_knode(pattern)
        ):
            if len(pattern_matches[pattern]) >= min_occurrences:
                common_patterns.append(pattern)
                seen_patterns.add(pattern)
        else:
            common_patterns.append(pattern)
            seen_patterns.add(pattern)

    # Step 4: Calculate bit gain and filter for positive savings
    def bit_gain(pat: KNode) -> float:
        count = pattern_counter[pat]
        # avg_len = sum(s.bit_length() for s in pattern_matches[pat]) / count
        current_len = sum(s.bit_length() for s in pattern_matches[pat])
        param_len = (
            sum(
                p.bit_length()
                for s in pattern_matches[pat]
                for _, ps in extract_template(s)
                if pat == _
                for p in ps
            )
            / count
        )
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        # return (count - 1) * (avg_len - symb_len) - pat.bit_length()
        return current_len - (count * symb_len + pat.bit_length())

    """
    # TO-DO: Currently repeats of simple moves by the same count of a move are not symbolized,
    # because the SymbolNode is heavier
    # so a lot of repeat by the same count won't be symbolize
    """
    common_patterns_test = [
        pat for pat in common_patterns if bit_gain(pat) > 1000
    ]
    for pat in common_patterns_test:
        print(f"\n\nPattern: {pat}")
        print(f"Bit gain: {bit_gain(pat)}")
        print(f"Count: {pattern_counter[pat]}")
        print(
            f"Current len: {sum(s.bit_length() for s in pattern_matches[pat])}"
        )
        print(
            f"Param len: {
                (
                    sum(
                        p.bit_length()
                        for s in pattern_matches[pat]
                        for _, ps in extract_template(s)
                        if pat == _
                        for p in ps
                    )
                    / pattern_counter[pat]
                )
            }"
        )
        param_len = (
            sum(
                p.bit_length()
                for s in pattern_matches[pat]
                for _, ps in extract_template(s)
                if pat == _
                for p in ps
            )
            / pattern_counter[pat]
        )
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        print(f"symb len: {symb_len}")
        print(f"pattern len: {pat.bit_length()}")

    # Might need
    # and pattern_counter[node] > tree_count (lattice_count)
    # You may want more than 1 per full tree ?
    common_patterns = [pat for pat in common_patterns if bit_gain(pat) > 0]

    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]


def symbolize_pattern(
    trees: Sequence[KNode[T]],
    symbols: Sequence[KNode[T]],
    new_symbol: KNode[T],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    index = IndexValue(len(symbols))
    trees = tuple(
        node_to_symbolized_node(index, new_symbol, tree) for tree in trees
    )
    symbols = tuple(
        node_to_symbolized_node(index, new_symbol, tree) for tree in symbols
    ) + (new_symbol,)
    return trees, symbols


def greedy_symbolization(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Symbolize trees by replacing the common pattern with SymbolNodes.
    It only do the most common pattern, because symbolizing one pattern potentially changes the bit length saving of the rest
    """

    # While there is a pattern whose abstraction leads to bit gain savings
    common_patterns = find_symbol_candidates(trees + symbols)
    while common_patterns:
        # Abstract the best one
        new_symbol = common_patterns[0]
        trees, symbols = symbolize_pattern(trees, symbols, new_symbol)
        common_patterns = find_symbol_candidates(trees + symbols)

    return (trees, symbols)


def symbolize(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    i = 0
    # Phase 1: Non-symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not is_symbolized(c) and not isinstance(c, RootNode)
        ]
        # print(f"candidates len: {len(candidates)}")
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])
        i += 1

    # # Phase 2: Symbolic patterns by depth
    # for depth in range(1, max_depth(trees) + 1):
    #     while True:
    #         candidates = [c for c in find_symbol_candidates(trees + symbols) if len(contained_symbols(c)) <= depth]
    #         if not candidates:
    #             break
    #         trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 2: Including Symbolic patterns:
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])
        i += 1

    # Phase 3: Include roots
    while True:
        candidates = find_symbol_candidates(
            trees + symbols
        )  # All candidates, including RootNodes
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    return trees, symbols


## Re-factorization:


def factor_by_existing_symbols(
    tree: KNode[T], symbols: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Factors a tree against an existing symbol table, replacing matches with SymbolNodes.

    Args:
        tree: The tree to factor.
        symbols: The symbol table to check against.

    Returns:
        Factored tree with applicable SymbolNode replacements.
    """

    def factor_node(node: KNode[T]) -> KNode[T]:
        for i, symbol in enumerate(symbols):
            abstracted = abstract_node(IndexValue(i), symbol, node)
            if abstracted != node:  # If abstraction occurred
                return abstracted
        return node

    return postmap(tree, factor_node, factorize=True)


def remap_symbol_indices(
    tree: KNode[T], mapping: list[int], tree_idx: int
) -> KNode[T]:
    """
    Updates SymbolNode indices in a tree based on the provided mapping.
    Used to remap the tree elements.

    Args:
        tree: The tree to update.
        mapping: List mapping old indices to new ones.
        tree_idx: Index of the tree for debugging purposes.

    Returns:
        Updated tree with remapped SymbolNode indices.
    """

    def update_node(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters)
        return node

    return postmap(tree, update_node, factorize=False)


def remap_sub_symbols(
    symbol: KNode[T], mapping: list[int], original_table: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Remaps SymbolNode indices within a symbol based on the provided mapping.
    Used to remap the symbol tables elements.

    Args:
        symbol: The symbol to update.
        mapping: List mapping old indices to new ones.
        original_table: The original symbol table for resolution if needed.

    Returns:
        Updated symbol with remapped SymbolNode indices.
    """

    def update_index(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters)
        return node

    return postmap(symbol, update_index, factorize=False)


def merge_symbol_tables(
    symbol_tables: Sequence[tuple[KNode[T], ...]],
) -> tuple[tuple[KNode[T], ...], list[list[int]]]:
    """
    Merges multiple symbol tables into a unified table, returning the table and mappings.
    It choose symbols so to minimize the total bit length.
    Note: Clean resymbolisation is often preferable.

    Args:
        symbol_tables: List of symbol tables to merge.

    Returns:
        Tuple of the unified symbol table and a list of mappings (old index -> new index) for each input table.
    """
    unified_symbols: list[KNode] = []
    mappings: list[list[int]] = [[] for _ in range(len(symbol_tables))]

    equivalence_classes: defaultdict[KNode, list[tuple[KNode, int]]] = (
        defaultdict(list)
    )  # Store symbol classes, in an union-find like structure. Each classes contains the symbols and their tables
    dependency_graph: defaultdict[tuple[KNode, int], set[int]] = defaultdict(
        set
    )  # Store for each symbol the symbols it depends on

    # Step 1: Collect all resolved symbolsa and build the dependency graph
    for i, table in enumerate(symbol_tables):
        for symbol in table:
            resolved_symbol = resolve_symbols(symbol, table)
            equivalence_classes[resolved_symbol].append((symbol, i))
            # Track dependencies: which symbols this symbol references in its table
            subsymbols = set(index.value for index in contained_symbols(symbol))
            dependency_graph[(symbol, i)] |= subsymbols

    # Step 2: Select optimal symbols per equivalence class
    selected_symbols = {}
    for resolved, symbols_in_class in equivalence_classes.items():
        # Prefer abstracted symbols (those with variables)
        abstracted = [s for s, _ in symbols_in_class if is_abstraction(s)]
        if abstracted:
            # Select the one with the most variables, then smallest bit_length
            selected = max(
                abstracted, key=lambda s: (arity(s), -s.bit_length())
            )
        else:
            # Select the one with the smallest bit_length
            selected = min(
                (s for s, _ in symbols_in_class), key=lambda s: s.bit_length()
            )
        selected_symbols[resolved] = selected

    # Step 3: Update dependency graph with selected symbols
    new_dependency_graph = defaultdict(set)
    symbol_to_selected = {}
    for resolved, selected in selected_symbols.items():
        for symbol, table_idx in equivalence_classes[resolved]:
            symbol_to_selected[(symbol, table_idx)] = selected
            table = symbol_tables[table_idx]
            for dep_idx in dependency_graph[(symbol, table_idx)]:
                if dep_idx < len(table):
                    dep_symbol = table[dep_idx]
                    dep_resolved = resolve_symbols(dep_symbol, table)
                    dep_selected = selected_symbols[dep_resolved]
                    new_dependency_graph[selected].add(dep_selected)

    # Step 4: Topological sort to order symbols in the unified table
    in_degree = {s: len(deps) for s, deps in new_dependency_graph.items()}
    for s in selected_symbols.values():
        if s not in in_degree:
            in_degree[s] = 0
    queue = deque([s for s, deg in in_degree.items() if deg == 0])
    unified_symbols = []

    while queue:
        symbol = queue.popleft()
        unified_symbols.append(symbol)
        for dependent in new_dependency_graph:
            if symbol in new_dependency_graph[dependent]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    symbol_to_index = {s: i for i, s in enumerate(unified_symbols)}
    mappings = [[] for _ in symbol_tables]

    for table_idx, table in enumerate(symbol_tables):
        for symbol in table:
            resolved = resolve_symbols(symbol, table)
            selected = selected_symbols[resolved]
            mappings[table_idx].append(symbol_to_index[selected])
    # TODO

    return tuple(unified_symbols), mappings


def symbolize_together(
    trees: tuple[KNode[T], ...], symbol_tables: Sequence[tuple[KNode[T], ...]]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Merges independently symbolized trees into a unified symbol table and re-symbolizes them together.

    Args:
        trees: Tuple of KNode[MoveValue] trees, each potentially containing SymbolNodes.
        symbol_tables: List of symbol tables corresponding to each tree (or empty if unsymbolized).

    Returns:
        Tuple of updated trees and the unified symbol table.
    """
    if symbol_tables:
        if len(symbol_tables) != len(trees):
            raise ValueError(
                f"There are only {len(symbol_tables)} symbol tables for {len(trees)} trees"
            )
        # Step 1: Merge symbol tables into a unified table with index remapping
        unified_symbols, mappings = merge_symbol_tables(symbol_tables)

        # Step 2: Update trees with the new symbol indices
        updated_trees = tuple(
            remap_symbol_indices(tree, mapping, i)
            for i, (tree, mapping) in enumerate(zip(trees, mappings))
        )

        # Step 3: Factor existing patterns against the unified symbol table
        factored_trees = tuple(
            factor_by_existing_symbols(tree, unified_symbols)
            for tree in updated_trees
        )
    else:
        factored_trees = trees
        unified_symbols = tuple()

    # Step 4: Find and symbolize new common patterns across all trees
    final_trees, final_symbols = symbolize(factored_trees, unified_symbols)

    return final_trees, final_symbols


def unsymbolize(knode: KNode[T], symbol_table: Sequence[KNode[T]]) -> KNode[T]:
    """
    Completely unsymbolize a given node.
    If the symbol table contains all the referenced templates, the resulting node
    should be free of any SymbolNodes or NestedNodes
    Second hypothesis: the symbol table has been completely resolved too.

    Args:
        knode: KNode containing NestedNodes and SymbolNodes.
        symbol_table: Symbol table containing all the templates referenced by NestedNodes and SymbolNodes
    """
    nnode = copy.deepcopy(knode)
    # Step 1: First unsymbolize SymbolNode
    nnode = resolve_symbols(nnode, symbol_table)

    # Step 2: Then unsymbolize NestedNodes
    nnode = expand_all_nested_nodes(nnode, symbol_table)

    return nnode


def unsymbolize_all(
    trees: Sequence[KNode[T]], symbol_table: Sequence[KNode[T]]
) -> tuple[KNode[T], ...]:
    """
    Hypothesis: no loop in the symbol table, a symbol can only reference a symbol strictly after him
    """
    # Step 1: Resolve the symbol table
    symbol_table = tuple(symbol_table)
    nsymbol_table = []
    for i, symb in enumerate(symbol_table):
        nsymbol_table.append(resolve_symbols(symb, symbol_table))

    nsymbol_table = tuple(nsymbol_table)

    # Step 2: Unsymbolize all the nodes
    return tuple(unsymbolize(tree, nsymbol_table) for tree in trees)


def full_symbolization(
    trees: Sequence[KNode[T]],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Full standard symbolization + node nesting.
    """
    # Step 1: nest nodes
    symbol_table = []
    nested = tuple(
        extract_nested_patterns(symbol_table, syntax_tree)
        for syntax_tree in trees
    )

    symbolized, symbol_table = symbolize(tuple(nested), tuple(symbol_table))

    return symbolized, symbol_table


