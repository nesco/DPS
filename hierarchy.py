from collections import defaultdict
from collections.abc import Iterator, Mapping, Set
from itertools import chain, combinations
from typing import cast

from typing_extensions import Iterable

from arc_syntax_tree import component_to_distribution
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
from localtypes import (
    Color,
    ColorGrid,
    Colors,
    Coord,
    Coords,
    GridObject,
)
from utils.dag_functionals import topological_sort
from utils.graph import nodes_to_connected_components
from utils.grid import (
    DIRECTIONS,
    grid_to_coords_by_color,
)


def coords_to_connected_components(coords: Coords) -> frozenset[Coords]:
    def node_to_neighbours(node: Coord) -> Coords:
        row, col = node
        possibilities = set(
            Coord(row + drow, col + dcol) for drow, dcol in DIRECTIONS.values()
        )
        neighbours = possibilities.intersection(coords)
        return neighbours

    return nodes_to_connected_components(coords, node_to_neighbours)


def condition_by_color_couples(
    grid: ColorGrid,
) -> dict[tuple[Color, Color], Coords]:
    """
    Returns a dict of all masks comprised of union of two colours.
    TO-DO: N colors?
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = sorted(list(coords_by_color.keys()))

    coords_by_color_couple: dict[tuple[Color, Color], Coords] = {}

    for i in range(len(colors)):
        color_a = colors[i]
        for j in range(i):
            color_b = colors[j]
            coords_by_color_couple[(color_a, color_b)] = (
                coords_by_color[color_a] | coords_by_color[color_b]
            )

        coords_by_color_couple[(color_a, color_a)] = coords_by_color[color_a]
    return coords_by_color_couple


def condition_by_colors(grid: ColorGrid) -> dict[Colors, Coords]:
    """
    Returns a dictionary where each key is a frozenset of colors (representing a subset of colors)
    and each value is the set of coordinates where the color in the grid is in that subset.
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = list(coords_by_color.keys())
    all_subsets = chain.from_iterable(
        combinations(colors, r) for r in range(1, len(colors) + 1)
    )
    result = {}
    for subset in all_subsets:
        S = frozenset(subset)
        union_coords = set().union(*(coords_by_color[c] for c in S))
        result[S] = union_coords
    return result


def grid_to_components_by_colors(
    grid: ColorGrid,
) -> defaultdict[Colors, set[Coords]]:
    components_by_colors = {}
    coords_by_colors = condition_by_colors(grid)
    for colors, coords in coords_by_colors.items():
        components_by_colors[colors] = coords_to_connected_components(coords)

    # Removing multiple of occurences of components associated with several color sets
    # A single colored component of color 1 like the cross of "2dc579da.json" will appear in the coords mask of {1} and {1, 3} for example
    colors_by_component = {}
    for colors, components in components_by_colors.items():
        for component in components:
            if component not in colors_by_component:
                colors_by_component[component] = colors
            else:
                if (
                    len(colors) < len(colors_by_component[component])
                ):  # Equivalent to <= here but sufficient, checking the len is quicker
                    colors_by_component[component] = colors

    # Now that each component is associated with its lowest count colors, the dict can be inverted again:
    new_components_by_colors: defaultdict[Colors, set] = defaultdict(set)
    for component, colors in colors_by_component.items():
        new_components_by_colors[colors].add(component)

    return new_components_by_colors


def components_by_colors_to_grid_object_dag(
    components_by_colors: Mapping[Colors, Set[Coords]],
) -> dict[GridObject, set[GridObject]]:
    """
    Transform the components into GridObjects, and build a DAG for the inclusion relation out of it, by building an Hass Diagram.
    In practice, the DAG acts a bit like a lattice, because selecting all the colors will end into a single connected component
    encompassing the entire grid.
    It will thus be the supremum of the inclusion relation, every other grid objects will be included into it.
    To form a full latice, the empty object with no colors nor coords should be included to, but will not for practicallity

    Args:
        components_by_colors: sets of components by the colors they are made of.

    Returns:
        dict[GridObject, set[GridObject]]: DAG of grid objects encoding the set inclusion relationship <=
    """

    # Step 1: Building Grid Objects
    grid_objects = tuple(
        GridObject(colors, component)
        for colors, components in components_by_colors.items()
        for component in components
    )

    # Step 2: Build the subsets and the supersets dict
    subsets = {
        a: frozenset(b for b in grid_objects if b.coords <= a.coords)
        for a in grid_objects
    }
    supersets = {
        b: frozenset(c for c in grid_objects if b.coords <= c.coords)
        for b in grid_objects
    }

    # Step 3: Initialize the DAG
    graph = {a: set() for a in grid_objects}

    # Step 4: For each set a, check all its proper subsets b
    for a in grid_objects:
        for b in subsets[a]:
            if b != a:  # Ensure b is a proper subset of a
                intersection = subsets[a] & supersets[b]
                if len(intersection) == 2:  # Only a and b, no intermediate c
                    graph[a].add(b)

    return graph


def sort_by_inclusion(
    grid_object_dag: Mapping[GridObject, Set[GridObject]],
) -> tuple[GridObject, ...]:
    return topological_sort(grid_object_dag)


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
    The final syntax trees are either:
        - RootNode with a single color
        - SumNode of RootNodes with a single colors
        - ProductNode of a RootNode with a single color and a previously described SumNode
    The last case is used for background normalization

    Returns:
        syntax_tree_by_object: dict[
                GridObject,
                RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue]
            ]
        sorted_grid_objects: tuple[GridObject, ...]

    """
    syntax_by_object: dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ] = {}

    # Step 1: Sort grid objects
    sorted_grid_objects = tuple(reversed(sort_by_inclusion(grid_object_dag)))

    # Step 2: Compute the best syntax tree for each component
    # And store it in its raw version
    # Note: I could do it for unicolored component only,
    # it would improve performance but degrade the quality of the cosymbolization
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
    # Replace the max length root node by a normalized version of the object syntax tree
    # if it has a greater bit length
    # If the set is of length one, take the only root node
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

        # Check if the object with the greastest bit length has a greater one than the object syntax tree
        # Ideally it should be corrected by the number of bits the colors of the object syntax tree takes
        max_index, max_bit_length = max(
            enumerate(bit_lengthes), key=lambda x: x[1]
        )

        if max_bit_length > symbolized[i].bit_length():
            # Create a ProductNode with first the object syntax tree with the color of the dependency to replace
            # then a SumNode (if needed) of the sub unicolored root nodes
            # (in practice a ProductNode would work too, but it's morea meaningful for it to be a SUmNode)
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
            # syntax_by_object[grid_object] = ProductNode((new_syntax_tree, sum))
        else:
            # Else a SumNode of the sub unicolored root nodes
            # (in practice a ProductNode would work too, but it's morea meaningful for it to be a SUmNode)
            sum = iterable_to_sum(unicolor_roots)
            assert isinstance(sum, SumNode | RootNode)
            normalized_syntax_trees.append(sum)
            # syntax_by_object[grid_object] = sum

    for i, object in enumerate(sorted_grid_objects):
        syntax_by_object[object] = normalized_syntax_trees[i]
    return syntax_by_object, sorted_grid_objects

    # Test Step 5: Cosymbolize them
    # for tree in normalized_syntax_trees:
    #     print(tree)

    # symbolized_syntax_trees, symbol_table = full_symbolization(
    #     normalized_syntax_trees
    # )

    # symbolyzed_syntax_by_object = {}
    # for i, st in enumerate(symbol_table):
    #     print(f"s_{i}: {st}")

    # for i, object in enumerate(sorted_grid_objects):
    #     syntax_by_object[object] = normalized_syntax_trees[i]
    #     symbolyzed_syntax_by_object[object] = symbolized_syntax_trees[i]

    # return syntax_by_object, symbol_table


def unpack_dependencies(
    syntax_trees: Iterable[
        ProductNode[MoveValue] | SumNode[MoveValue] | RootNode[MoveValue]
    ],
) -> Iterator[ProductNode[MoveValue] | RootNode[MoveValue]]:
    for st in syntax_trees:
        if isinstance(st, SumNode):
            children = cast(
                frozenset[ProductNode[MoveValue] | RootNode[MoveValue]],
                st.children,
            )
            yield from children
        else:
            yield st


def dag_to_syntax_trees(grid_object_dag: Mapping[GridObject, Set[GridObject]]):
    # Step 1: Sort grid objects by topological order
    sorted_grid_objects = tuple(reversed(sort_by_inclusion(grid_object_dag)))

    # Step 2: Get the minimal multi-colored syntax tree per grid object
    # in it's raw form
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
    # track replaced syntax trees, as they can be part of several objects
    # replaced = set() # The issue is there, it can lead to vanishing objects

    for object in sorted_grid_objects:
        replaced = set() # TODO: Test this extensively
        # Get dependencies
        dependencies = grid_object_dag[object]
        current_root = root_by_grid_object[object]

        # If an object has no dependency (or one? it can't have just one), it's syntax tree is its standard root
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

        # Else, retrieve all the dependencies syntax trees
        dependencies_syntax_trees = set(
            syntax_tree_by_grid_object[dependency]
            for dependency in dependencies
        )

        # Unpack them
        # set because a given syntax tree can be part of several dependencies
        syntax_trees = tuple(
            st
            for st in unpack_dependencies(dependencies_syntax_trees)
            if st not in replaced
        )

        # Keep only the backround of product nodes before symbolizing them
        to_symbolize = tuple(
            st.children[0] if isinstance(st, ProductNode) else st
            for st in syntax_trees
        )

        # Symbolize them with the current object syntaxt tree
        symbolized, symbol_table = full_symbolization(
            to_symbolize + (root_by_grid_object[object],)
        )

        # Get the index of the syntax tree of maximum bit length:
        max_index, max_bit_length = max(
            enumerate(symbolized), key=lambda x: x[1].bit_length()
        )

        # If it's the multicolored root of the current object
        # Take a SumNode of the dependencies
        if max_index == len(syntax_trees):
            sum = iterable_to_sum(
                syntax_trees
            )  # It should always be > 1 so no need for the iterable. Just in case one day
            assert isinstance(sum, SumNode)
            syntax_tree_by_grid_object[object] = sum
            continue

        # Else, replace swap largest syntax tree
        # with the syntax tree of the object with the color of the largest unicolored root or background instead
        max_dependency = syntax_trees[max_index]
        replaced.add(max_dependency)
        ndependencies = syntax_trees[:max_index] + syntax_trees[max_index + 1 :]

        match max_dependency:
            case RootNode(_, _, colors):
                color = colors
                sum = iterable_to_sum(ndependencies)
            case ProductNode(children):
                # Swap the background
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
        print(f"Choice: {ProductNode((new_root, sum))}")
        syntax_tree_by_grid_object[object] = ProductNode((new_root, sum))

    print("\n\n")
    for go in sorted_grid_objects:
        print(f"{syntax_tree_by_grid_object[go]}")

    return syntax_tree_by_grid_object, sorted_grid_objects


def test_coords_to_connected_components():
    test_cases = [
        (set(), frozenset()),
        ({(0, 0)}, frozenset([frozenset({(0, 0)})])),
        (
            {(0, 0), (1, 0), (0, 1)},
            frozenset([frozenset({(0, 0), (1, 0), (0, 1)})]),
        ),
        (
            {(0, 0), (2, 2)},
            frozenset([frozenset({(0, 0)}), frozenset({(2, 2)})]),
        ),
        (
            {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)},
            frozenset(
                [
                    frozenset({(0, 0), (0, 1), (0, 2)}),
                    frozenset({(2, 0), (2, 1), (2, 2)}),
                ]
            ),
        ),
        # New test cases for 8-connectivity
        ({(0, 0), (1, 1)}, frozenset([frozenset({(0, 0), (1, 1)})])),
        (
            {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)},
            frozenset([frozenset({(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)})]),
        ),
    ]
    for coords, expected in test_cases:
        result = coords_to_connected_components(coords)
        assert result == expected, (
            f"Failed for coords={coords}: expected {expected}, got {result}"
        )


def test_condition_by_color_couples():
    grid1 = [[0, 1], [2, 0]]
    expected1 = {
        (0, 0): {(0, 0), (1, 1)},
        (1, 0): {(0, 0), (1, 0), (1, 1)},
        (1, 1): {(1, 0)},
        (2, 0): {(0, 0), (0, 1), (1, 1)},
        (2, 1): {(0, 1), (1, 0)},
        (2, 2): {(0, 1)},
    }
    result1 = condition_by_color_couples(grid1)
    assert result1 == expected1, (
        f"Failed for grid1: expected {expected1}, got {result1}"
    )

    grid2 = [[0, 0], [0, 0]]
    expected2 = {(0, 0): {(0, 0), (1, 0), (0, 1), (1, 1)}}
    result2 = condition_by_color_couples(grid2)
    assert result2 == expected2, (
        f"Failed for grid2: expected {expected2}, got {result2}"
    )

    grid3 = [[0, 1, 2], [2, 1, 0]]
    expected3 = {
        (0, 0): {(0, 0), (2, 1)},
        (1, 0): {(0, 0), (1, 0), (1, 1), (2, 1)},
        (1, 1): {(1, 0), (1, 1)},
        (2, 0): {(0, 0), (0, 1), (2, 0), (2, 1)},
        (2, 1): {(0, 1), (1, 0), (1, 1), (2, 0)},
        (2, 2): {(2, 0), (0, 1)},
    }
    result3 = condition_by_color_couples(grid3)
    assert result3 == expected3, (
        f"Failed for grid3: expected {expected3}, got {result3}"
    )


def test_condition_by_colors():
    grid1 = [[0, 1], [2, 0]]
    result1 = condition_by_colors(grid1)
    assert result1[frozenset({0})] == {(0, 0), (1, 1)}
    assert result1[frozenset({1})] == {(1, 0)}
    assert result1[frozenset({2})] == {(0, 1)}
    assert result1[frozenset({0, 1})] == {(0, 0), (1, 0), (1, 1)}
    assert result1[frozenset({0, 2})] == {(0, 0), (0, 1), (1, 1)}
    assert result1[frozenset({1, 2})] == {(1, 0), (0, 1)}
    assert result1[frozenset({0, 1, 2})] == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(result1) == 7  # 2^3 - 1
    print("Tests passed!")


def test_grid_to_components_by_colors():
    """
    Test function for grid_to_components_by_colors with 8-connectivity.
    Tests various grid configurations to ensure correct component identification.
    """
    # Test Case 1: 2x2 grid with three colors
    grid1 = [[0, 1], [2, 0]]
    result1 = grid_to_components_by_colors(grid1)
    expected1 = defaultdict(
        set,
        {
            frozenset({0}): {frozenset({Coord(0, 0), Coord(1, 1)})},
            frozenset({1}): {frozenset({Coord(1, 0)})},
            frozenset({2}): {frozenset({Coord(0, 1)})},
            frozenset({0, 1}): {
                frozenset({Coord(0, 0), Coord(1, 0), Coord(1, 1)})
            },
            frozenset({0, 2}): {
                frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 1)})
            },
            frozenset({1, 2}): {frozenset({Coord(1, 0), Coord(0, 1)})},
            frozenset({0, 1, 2}): {
                frozenset({Coord(0, 0), Coord(1, 0), Coord(0, 1), Coord(1, 1)})
            },
        },
    )
    assert set(result1.keys()) == set(expected1.keys()), (
        "Keys mismatch in Test Case 1"
    )
    for key in expected1:
        assert result1[key] == expected1[key], (
            f"Mismatch for key {key} in Test Case 1: expected {expected1[key]}, got {result1[key]}"
        )

    # Test Case 2: 1x1 grid with single color
    grid2 = [[0]]
    result2 = grid_to_components_by_colors(grid2)
    expected2 = defaultdict(set, {frozenset({0}): {frozenset({Coord(0, 0)})}})
    assert set(result2.keys()) == set(expected2.keys()), (
        "Keys mismatch in Test Case 2"
    )
    for key in expected2:
        assert result2[key] == expected2[key], (
            f"Mismatch for key {key} in Test Case 2: expected {expected2[key]}, got {result2[key]}"
        )

    # Test Case 3: 2x2 grid with two colors and diagonal connectivity
    grid3 = [[0, 1], [1, 0]]
    result3 = grid_to_components_by_colors(grid3)
    expected3 = defaultdict(
        set,
        {
            frozenset({0}): {frozenset({Coord(0, 0), Coord(1, 1)})},
            frozenset({1}): {frozenset({Coord(0, 1), Coord(1, 0)})},
            frozenset({0, 1}): {
                frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)})
            },
        },
    )
    assert set(result3.keys()) == set(expected3.keys()), (
        "Keys mismatch in Test Case 3"
    )
    for key in expected3:
        assert result3[key] == expected3[key], (
            f"Mismatch for key {key} in Test Case 3: expected {expected3[key]}, got {result3[key]}"
        )

    # Test Case 4: 2x2 grid with single color (fully connected)
    grid4 = [[0, 0], [0, 0]]
    result4 = grid_to_components_by_colors(grid4)
    expected4 = defaultdict(
        set,
        {
            frozenset({0}): {
                frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)})
            }
        },
    )
    assert set(result4.keys()) == set(expected4.keys()), (
        "Keys mismatch in Test Case 4"
    )
    for key in expected4:
        assert result4[key] == expected4[key], (
            f"Mismatch for key {key} in Test Case 4: expected {expected4[key]}, got {result4[key]}"
        )

    print("All tests for grid_to_components_by_colors passed!")


def test_components_by_colors_to_grid_object_dag():
    # Define the input
    components_by_colors: dict[Colors, frozenset[Coords]] = {
        frozenset({0}): frozenset({frozenset({Coord(0, 0)})}),
        frozenset({1}): frozenset({frozenset({Coord(1, 1)})}),
        frozenset({2}): frozenset({frozenset({Coord(2, 2)})}),
        frozenset({0, 1}): frozenset({frozenset({Coord(0, 0), Coord(1, 1)})}),
        frozenset({0, 2}): frozenset({frozenset({Coord(0, 0), Coord(2, 2)})}),
        frozenset({1, 2}): frozenset({frozenset({Coord(1, 1), Coord(2, 2)})}),
        frozenset({0, 1, 2}): frozenset(
            {frozenset({Coord(0, 0), Coord(1, 1), Coord(2, 2)})}
        ),
    }

    # Compute the result
    result = components_by_colors_to_grid_object_dag(components_by_colors)

    # Define GridObjects for the expected output
    GO0 = GridObject(frozenset({0}), frozenset({Coord(0, 0)}))
    GO1 = GridObject(frozenset({1}), frozenset({Coord(1, 1)}))
    GO2 = GridObject(frozenset({2}), frozenset({Coord(2, 2)}))
    GO01 = GridObject(frozenset({0, 1}), frozenset({Coord(0, 0), Coord(1, 1)}))
    GO02 = GridObject(frozenset({0, 2}), frozenset({Coord(0, 0), Coord(2, 2)}))
    GO12 = GridObject(frozenset({1, 2}), frozenset({Coord(1, 1), Coord(2, 2)}))
    GO012 = GridObject(
        frozenset({0, 1, 2}), frozenset({Coord(0, 0), Coord(1, 1), Coord(2, 2)})
    )

    # Define the expected DAG
    expected_graph = {
        GO012: {GO01, GO02, GO12},
        GO01: {GO0, GO1},
        GO02: {GO0, GO2},
        GO12: {GO1, GO2},
        GO0: set(),
        GO1: set(),
        GO2: set(),
    }

    # Verify the result matches the expected graph
    assert set(result.keys()) == set(expected_graph.keys()), (
        "DAG keys do not match expected keys"
    )
    for key in expected_graph:
        assert result[key] == expected_graph[key], (
            f"DAG mismatch for {key}: expected {expected_graph[key]}, got {result[key]}"
        )


def grid_to_syntax_trees(
    grid: ColorGrid,
) -> tuple[
    dict[
        GridObject,
        RootNode[MoveValue] | SumNode[MoveValue] | ProductNode[MoveValue],
    ],
    tuple[GridObject, ...],
]:
    """
    Chains together the following functions:
    1. grid_to_components_by_colors: Converts a grid to components by colors
    2. components_by_colors_to_grid_object_dag: Converts components to a DAG of grid objects
    3. dag_to_syntax_trees: Converts the DAG to syntax trees

    Args:
        grid: A ColorGrid representing the input grid

    Returns:
        tuple containing:
        - A dictionary mapping GridObjects to their syntax trees
        - A tuple of sorted GridObjects
    """
    # Step 1: Convert grid to components by colors
    components_by_colors = grid_to_components_by_colors(grid)

    # Step 2: Convert components to a DAG of grid objects
    grid_object_dag = components_by_colors_to_grid_object_dag(
        components_by_colors
    )
    # Step 3: Convert the DAG to syntax trees
    syntax_trees, sorted_grid_objects = dag_to_syntax_trees(grid_object_dag)

    return syntax_trees, sorted_grid_objects


if __name__ == "__main__":
    test_coords_to_connected_components()
    test_condition_by_color_couples()
    test_condition_by_colors()
    test_grid_to_components_by_colors()
    test_components_by_colors_to_grid_object_dag()

    print("All tests passed!")
