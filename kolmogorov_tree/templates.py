"""
Template extraction for Kolmogorov Tree.

Functions:
    extract_template(node) - Generate abstracted versions with variables

    Recursive Pattern Detection:
        extract_nested_sum_template(node)     - Find nested pattern in SumNode
        extract_nested_product_template(node) - Find nested pattern in ProductNode
        detect_recursive_collection(node)     - Detect recursive nesting pattern
        nested_collection_to_nested_node(node) - Convert pattern to NestedNode

Types:
    Parameters = tuple[BitLengthAware, ...]  - Values to substitute for variables
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
    """
    Generates abstracted versions of a node by replacing subparts with variables.

    Returns list of (abstracted_node, parameters) pairs.
    Does not abstract nodes that already contain variables.
    """
    abstractions: list[tuple[KNode[T], Parameters]] = []

    if is_abstraction(knode):
        return abstractions

    match knode:
        case ProductNode(children) if len(children) > 2:
            child_counter = Counter(children)
            child_set = list(child_counter.keys())
            top_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]

            # Abstract most frequent child
            nodes1 = tuple(
                VariableNode(VariableValue(0)) if c == top_children[0] else c
                for c in children
            )
            abstractions.append((ProductNode(nodes1), (top_children[0],)))

            if len(top_children) > 1 and len(children) > 2:
                # Abstract second most frequent
                nodes2 = tuple(
                    VariableNode(VariableValue(0)) if c == top_children[1] else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes2), (top_children[0],)))

                # Abstract both
                nodes3 = tuple(
                    VariableNode(VariableValue(top_children.index(c)))
                    if c in top_children
                    else c
                    for c in children
                )
                abstractions.append((ProductNode(nodes3), tuple(top_children)))

        case SumNode(children) if len(children) > 2:
            child_counter = Counter(children)
            child_set = list(child_counter.keys())
            top_children = sorted(
                child_set,
                key=lambda x: x.bit_length() * child_counter[x],
                reverse=True,
            )[:2]

            nodes1 = frozenset(
                VariableNode(VariableValue(0)) if c == top_children[0] else c
                for c in children
            )
            abstractions.append((SumNode(nodes1), (top_children[0],)))

            if len(top_children) > 1 and len(children) > 2:
                nodes2 = frozenset(
                    VariableNode(VariableValue(0)) if c == top_children[1] else c
                    for c in children
                )
                abstractions.append((SumNode(nodes2), (top_children[0],)))

                nodes3 = frozenset(
                    VariableNode(VariableValue(top_children.index(c)))
                    if c in top_children
                    else c
                    for c in children
                )
                abstractions.append((SumNode(nodes3), tuple(top_children)))

        case RepeatNode(node, count):
            abstractions.extend(
                [
                    (RepeatNode(VariableNode(VariableValue(0)), count), (node,)),
                    (RepeatNode(node, VariableNode(VariableValue(0))), (count,)),
                ]
            )

        case NestedNode(index, node, count):
            abstractions.extend(
                [
                    (NestedNode(index, VariableNode(VariableValue(0)), count), (node,)),
                    (NestedNode(index, node, VariableNode(VariableValue(0))), (count,)),
                    (
                        NestedNode(
                            index,
                            VariableNode(VariableValue(0)),
                            VariableNode(VariableValue(1)),
                        ),
                        (node, count),
                    ),
                ]
            )

        case RootNode(node, position, colors):
            abstractions.extend(
                [
                    (
                        RootNode(VariableNode(VariableValue(0)), position, colors),
                        (node,),
                    ),
                    (
                        RootNode(node, VariableNode(VariableValue(0)), colors),
                        (position,),
                    ),
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
            if len(colors.value) == 1:  # type: ignore
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
                    (RectNode(VariableNode(VariableValue(0)), width), (height,)),
                    (RectNode(height, VariableNode(VariableValue(0))), (width,)),
                ]
            )

        case _:
            pass

    return abstractions


def extract_nested_sum_template(
    snode: SumNode[T],
) -> tuple[SumNode[T], SumNode[T]] | None:
    """
    Finds nested pattern in SumNode: inner ProductNode containing a SumNode.

    Returns (template_with_variable, inner_sum) or None.
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
            VariableNode(VariableValue(0)) if node == max_sum else node
            for node in max_product.children
        )
    )

    new_sum = SumNode(
        frozenset(
            new_product if node == max_product else node for node in snode.children
        )
    )

    return new_sum, max_sum


def extract_nested_product_template(
    pnode: ProductNode[T],
) -> tuple[ProductNode[T], ProductNode[T]] | None:
    """
    Finds nested pattern in ProductNode: inner SumNode containing a ProductNode.

    Returns (template_with_variable, inner_product) or None.
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
            VariableNode(VariableValue(0)) if node == max_product else node
            for node in max_sum.children
        )
    )

    new_product = ProductNode(
        tuple(new_sum if node == max_sum else node for node in pnode.children)
    )

    return new_product, max_product


def detect_recursive_collection(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[KNode[T], KNode[T], int] | None:
    """
    Detects repeated nesting pattern in a collection node.

    Returns (template, terminal_node, recursion_depth) or None.
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
            common_template = template
        elif template != common_template:
            break

        if not isinstance(parameter, type(knode)):
            break

        count += 1
        current = parameter

    if count >= 1 and common_template is not None:
        return common_template, current, count
    return None


def nested_collection_to_nested_node(
    knode: SumNode[T] | ProductNode[T],
) -> tuple[NestedNode[T], KNode[T]] | None:
    """
    Converts a recursively nested collection to a NestedNode.

    Returns (nested_node, template) or None if no pattern found.
    The returned NestedNode has index=0; caller must update with actual symbol table index.
    """
    recursive_info = detect_recursive_collection(knode)
    if recursive_info is None:
        return None

    template, terminal_node, count = recursive_info
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
