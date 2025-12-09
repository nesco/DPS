"""
Template extraction utilities for Kolmogorov Tree.

This module provides functions for extracting templates (abstractions) from KNodes
by replacing subparts with variables, as well as detecting and transforming
recursive patterns in collection nodes.
"""

from collections import Counter

from localtypes import BitLengthAware

from kolmogorov_tree.nodes import (
    KNode,
    NestedNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    VariableNode,
)
from kolmogorov_tree.predicates import is_abstraction
from kolmogorov_tree.primitives import (
    CountValue,
    IndexValue,
    T,
    VariableValue,
)

type Parameters = tuple[BitLengthAware, ...]


def extract_template(knode: KNode[T]) -> list[tuple[KNode[T], Parameters]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions: list[tuple[KNode[T], Parameters]] = []

    # If the node is already a lambda-abstraction
    # It should not be abstracted further
    if is_abstraction(knode):
        return abstractions

    match knode:
        case ProductNode(children) if len(children) > 2:  # noqa: F811
            # Abstract up to two distinct elements for now
            child_counter = Counter(children)  # Count occurrences of each child
            child_set = list(child_counter.keys())
            max_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]  # I don't know why it works better when it's 4 here instead of 2

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
                    VariableNode(VariableValue(0)) if c == max_children[1] else c
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
                    VariableNode(VariableValue(0)) if c == max_children[1] else c
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
                        NestedNode(index, VariableNode(VariableValue(0)), count),
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
                        RootNode(VariableNode(VariableValue(0)), position, colors),
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
        case RectNode(height, width):
            if height == width:
                abstractions.append(
                    (
                        RectNode(
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(0)),
                        ),
                        (height,),
                    )
                )
                abstractions.extend(
                    [
                        (
                            RectNode(VariableNode(VariableValue(0)), width),
                            (height,),
                        ),
                        (
                            RectNode(height, VariableNode(VariableValue(0))),
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
            {node if node != max_product else new_product for node in snode.children}
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
            common_template = template  # Set the template on the first iteration
        elif template != common_template:
            break  # Stop if the template changes

        if not isinstance(parameter, type(knode)):
            break  # Stop if the parameter isn't a SumNode

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


__all__ = [
    "Parameters",
    "extract_template",
    "extract_nested_sum_template",
    "extract_nested_product_template",
    "detect_recursive_collection",
    "nested_collection_to_nested_node",
]
