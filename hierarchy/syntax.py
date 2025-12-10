"""
Syntax tree construction from grid object DAGs.

Converts the inclusion DAG into Kolmogorov syntax trees with:
- Background normalization (larger structures replace smaller ones)
- Co-symbolization across related objects
"""

import logging
from collections.abc import Iterator, Mapping, Set
from typing import cast

from typing_extensions import Iterable

from arc import component_to_distribution
from kolmogorov_tree import (
    MoveValue,
    PaletteValue,
    ProductNode,
    RootNode,
    SumNode,
    full_symbolization,
    iterable_to_sum,
    unsymbolize,
)
from localtypes import GridObject

from .dag import sort_by_inclusion

logger = logging.getLogger(__name__)


def unpack_dependencies(
    syntax_trees: Iterable[
        ProductNode[MoveValue] | SumNode[MoveValue] | RootNode[MoveValue]
    ],
) -> Iterator[ProductNode[MoveValue] | RootNode[MoveValue]]:
    """
    Unpack syntax trees, flattening SumNodes to their children.

    Args:
        syntax_trees: Iterable of syntax trees.

    Yields:
        Individual RootNodes or ProductNodes from the input trees.
    """
    for st in syntax_trees:
        if isinstance(st, SumNode):
            children = cast(
                frozenset[ProductNode[MoveValue] | RootNode[MoveValue]],
                st.children,
            )
            yield from children
        else:
            yield st


def dag_to_syntax_trees_linear(
    grid_object_dag: Mapping[GridObject, Set[GridObject]],
) -> tuple[
    dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ],
    tuple[GridObject, ...],
]:
    """
    Convert DAG to syntax trees using linear symbolization.

    Note: This is an older implementation. Consider using dag_to_syntax_trees instead.

    The final syntax trees are either:
        - RootNode with a single color
        - SumNode of RootNodes with single colors
        - ProductNode of a RootNode with a single color and a SumNode

    The last case is used for background normalization.

    Args:
        grid_object_dag: The inclusion DAG.

    Returns:
        Tuple of (syntax_tree_by_object dict, sorted_grid_objects tuple).
    """
    syntax_by_object: dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ] = {}

    # Step 1: Sort grid objects
    sorted_grid_objects = tuple(reversed(sort_by_inclusion(grid_object_dag)))

    # Step 2: Compute the best syntax tree for each component
    syntax_trees: list[RootNode[MoveValue]] = []

    for grid_object in sorted_grid_objects:
        distribution, symbol_table = component_to_distribution(
            grid_object.coords, grid_object.colors
        )
        root = unsymbolize(distribution[0], symbol_table)
        assert isinstance(root, RootNode)
        syntax_trees.append(root)

    # Step 2: Symbolize them together
    symbolized, symbol_table = full_symbolization(syntax_trees)

    # Step 3: Construct a flat representation first
    flat_syntax_by_object: dict[
        GridObject,
        set[RootNode[MoveValue]],
    ] = {}

    # If an object has no sub-object add its unicolor syntax tree
    # Otherwise add the dependencies syntax trees
    for i, grid_object in enumerate(sorted_grid_objects):
        dependencies = grid_object_dag[grid_object]
        if not dependencies:
            flat_syntax_by_object[grid_object] = set({syntax_trees[i]})
        else:
            flat_syntax_by_object[grid_object] = set.union(
                *[
                    flat_syntax_by_object[dependency]
                    for dependency in dependencies
                ]
            )

    # Step 4: For each object, construct the final object
    normalized_syntax_trees: list[
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue]
    ] = []
    for i, grid_object in enumerate(sorted_grid_objects):
        unicolor_roots = list(flat_syntax_by_object[grid_object])
        bit_lengthes = [
            symbolized[syntax_trees.index(root)].bit_length()
            for root in unicolor_roots
        ]

        syntax_tree = syntax_trees[i]

        max_index, max_bit_length = max(
            enumerate(bit_lengthes), key=lambda x: x[1]
        )

        if max_bit_length > symbolized[i].bit_length():
            new_syntax_tree = RootNode(
                syntax_tree.node,
                syntax_tree.position,
                unicolor_roots[max_index].colors,
            )
            sum = iterable_to_sum(
                unicolor_roots[:max_index] + unicolor_roots[max_index + 1 :]
            )
            assert sum is not None

            normalized_syntax_trees.append(ProductNode((new_syntax_tree, sum)))
        else:
            sum = iterable_to_sum(unicolor_roots)
            assert isinstance(sum, SumNode | RootNode)
            normalized_syntax_trees.append(sum)

    for i, object in enumerate(sorted_grid_objects):
        syntax_by_object[object] = normalized_syntax_trees[i]
    return syntax_by_object, sorted_grid_objects


def dag_to_syntax_trees(
    grid_object_dag: Mapping[GridObject, Set[GridObject]],
) -> tuple[
    dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ],
    tuple[GridObject, ...],
]:
    """
    Convert the inclusion DAG to syntax trees with background normalization.

    For each grid object:
    1. Compute its multi-colored syntax tree
    2. Collect syntax trees from dependencies
    3. Decide whether to use the object's tree as background or keep dependency trees

    Background normalization: If the object's syntax tree is more complex than
    the largest dependency, it becomes the background with the dependency's color.

    Args:
        grid_object_dag: The inclusion DAG.

    Returns:
        Tuple of (syntax_tree_by_object dict, sorted_grid_objects tuple).
    """
    # Step 1: Sort grid objects by topological order
    sorted_grid_objects = tuple(reversed(sort_by_inclusion(grid_object_dag)))

    # Step 2: Get the minimal multi-colored syntax tree per grid object
    root_by_grid_object: dict[GridObject, RootNode[MoveValue]] = dict()

    for grid_object in sorted_grid_objects:
        distribution, symbol_table = component_to_distribution(
            grid_object.coords, grid_object.colors
        )
        root = unsymbolize(distribution[0], symbol_table)
        assert isinstance(root, RootNode)
        root_by_grid_object[grid_object] = root

    # Step 3: Construct the unicolored syntax tree for each object
    syntax_tree_by_grid_object: dict[
        GridObject,
        ProductNode[MoveValue] | SumNode[MoveValue] | RootNode[MoveValue],
    ] = dict()

    for object in sorted_grid_objects:
        replaced = set()
        dependencies = grid_object_dag[object]
        current_root = root_by_grid_object[object]

        # If an object has no dependency, its syntax tree is its standard root
        if not dependencies:
            if not isinstance(current_root.colors, PaletteValue):
                raise ValueError(
                    f"Root has no palette value: {current_root.colors}"
                )
            if len(current_root.colors.value) != 1:
                raise ValueError(
                    f"Object without dependency: {object} has a multicolored root: {current_root.colors.value}"
                )
            syntax_tree_by_grid_object[object] = current_root
            continue

        # Retrieve all dependencies syntax trees
        dependencies_syntax_trees = set(
            syntax_tree_by_grid_object[dependency]
            for dependency in dependencies
        )

        # Unpack them (flatten SumNodes)
        syntax_trees = tuple(
            st
            for st in unpack_dependencies(dependencies_syntax_trees)
            if st not in replaced
        )

        # Keep only the background of product nodes before symbolizing
        to_symbolize = tuple(
            st.children[0] if isinstance(st, ProductNode) else st
            for st in syntax_trees
        )

        # Symbolize with the current object syntax tree
        symbolized, symbol_table = full_symbolization(
            to_symbolize + (root_by_grid_object[object],)
        )

        # Get the index of the syntax tree with maximum bit length
        max_index, max_bit_length = max(
            enumerate(symbolized), key=lambda x: x[1].bit_length()
        )

        # If it's the multicolored root of the current object
        # Take a SumNode of the dependencies
        if max_index == len(syntax_trees):
            sum = iterable_to_sum(syntax_trees)
            assert isinstance(sum, SumNode)
            syntax_tree_by_grid_object[object] = sum
            continue

        # Else, swap largest syntax tree with the object's syntax tree
        # using the color of the largest unicolored root or background
        max_dependency = syntax_trees[max_index]
        replaced.add(max_dependency)
        ndependencies = syntax_trees[:max_index] + syntax_trees[max_index + 1 :]

        match max_dependency:
            case RootNode(_, _, colors):
                color = colors
                sum = iterable_to_sum(ndependencies)
            case ProductNode(children):
                first = children[0]
                assert isinstance(children[1], SumNode | RootNode)
                if not isinstance(first, RootNode):
                    raise ValueError(
                        "Top level ProductNode doesn't begin by a RootNode setting the background color"
                    )
                color = first.colors
                sum = iterable_to_sum(
                    ndependencies + tuple(unpack_dependencies([children[1]]))
                )
            case _:
                raise TypeError(f"Invalid syntax tree: {max_dependency}")

        new_root = RootNode(current_root.node, current_root.position, color)
        assert sum is not None
        logger.debug(f"Background swap: {ProductNode((new_root, sum))}")
        syntax_tree_by_grid_object[object] = ProductNode((new_root, sum))

    # Log final syntax trees
    logger.debug("Final syntax trees:")
    for go in sorted_grid_objects:
        logger.debug(f"  {syntax_tree_by_grid_object[go]}")

    return syntax_tree_by_grid_object, sorted_grid_objects
