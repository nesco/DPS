from collections.abc import Sequence

from arc_syntax_tree import (
    component_to_raw_syntax_tree_distribution,
    decode_root,
    syntax_tree_at,
)
from hierarchy import grid_to_components_by_colors
from kolmogorov_tree import (
    KNode,
    MoveValue,
    breadth_first_preorder_knode,
    expand_all_nested_nodes,
    extract_nested_patterns,
    find_symbol_candidates,
    root_to_symbolize,
)
from localtypes import Colors, Coord, Coords, Proportions
from utils.display import display_raw_distribution
from utils.grid import (
    PointsOperations,
    coords_to_points,
    points_to_coords,
)
from utils.loader import train_task_to_grids


def test_reconstruction():
    inputs, outputs, input_test, output_test = train_task_to_grids(
        "2dc579da.json"
    )
    input = inputs[2]

    components_by_colors = grid_to_components_by_colors(input)

    cross = max(components_by_colors[frozenset({1})], key=len)
    a_rect = max(components_by_colors[frozenset({3})], key=len)
    rect_with_hole = next(iter(components_by_colors[frozenset({1, 3})]))

    def check_coords_compositional_identity(
        component: Coords, colors: Colors, start: Coord
    ):
        syntax_tree = syntax_tree_at(component, colors, start)
        coords = points_to_coords(decode_root(syntax_tree))
        assert component == coords, (
            f"Compositional identity for `syntax_tree_at` and `decode_root` violated for {component} with decoding giving: {coords}"
        )

    def check_coords_compositional_identity_distribution(
        component: Coords, colors: Colors, start: Coord
    ):
        syntax_trees = component_to_raw_syntax_tree_distribution(
            component, colors
        )
        for i, st in enumerate(syntax_trees):
            coords = points_to_coords(decode_root(st))
            assert component == coords, f"Test n°{i}, St: {
                st
            }, Compositional identity for `syntax_tree_at` and `decode_root` violated for {
                component
            } with decoding giving: {coords}. Differences: {
                component.difference(coords)
            } {coords.difference(component)}, {
                PointsOperations.print(
                    coords_to_points(component), Proportions(11, 11)
                )
            }, {
                PointsOperations.print(
                    coords_to_points(coords), Proportions(11, 11)
                )
            }"

    # Inputing fake unicolors so can be decoded
    # Test Case 1: Simple start point
    check_coords_compositional_identity(
        rect_with_hole, frozenset({1}), next(iter(rect_with_hole))
    )

    check_coords_compositional_identity(
        cross, frozenset({1}), next(iter(cross))
    )
    check_coords_compositional_identity(
        a_rect, frozenset({1}), next(iter(a_rect))
    )

    # Test Case 2: Whole distribution
    check_coords_compositional_identity_distribution(
        rect_with_hole, frozenset({1}), next(iter(rect_with_hole))
    )

    check_coords_compositional_identity_distribution(
        cross, frozenset({1}), next(iter(cross))
    )
    check_coords_compositional_identity_distribution(
        a_rect, frozenset({1}), next(iter(a_rect))
    )


def test_nested_nodes():
    inputs, outputs, input_test, output_test = train_task_to_grids(
        "2dc579da.json"
    )
    input = inputs[2]

    components_by_colors = grid_to_components_by_colors(input)

    cross = max(components_by_colors[frozenset({1})], key=len)
    rect = max(components_by_colors[frozenset({3})], key=len)
    rect_with_hole = next(iter(components_by_colors[frozenset({1, 3})]))

    cross_distribution = component_to_raw_syntax_tree_distribution(
        cross, frozenset({1})
    )

    rect_distribution = component_to_raw_syntax_tree_distribution(
        rect, frozenset({3})
    )
    rect_with_hole_distribution = component_to_raw_syntax_tree_distribution(
        rect_with_hole, frozenset({1})
    )

    def check_nested_node_identity(
        raw_st_distribution: Sequence[KNode[MoveValue]],
    ):
        symbol_table = []
        compressed_st_distribution = tuple(
            extract_nested_patterns(symbol_table, raw_syntax_tree)
            for raw_syntax_tree in raw_st_distribution
        )
        uncompressed_st_distribution = tuple(
            expand_all_nested_nodes(compressed_st, symbol_table)
            for compressed_st in compressed_st_distribution
        )
        assert uncompressed_st_distribution == raw_st_distribution

    check_nested_node_identity(cross_distribution)
    check_nested_node_identity(rect_distribution)
    check_nested_node_identity(rect_with_hole_distribution)

    print("NestedNode compression and uncompression are reversed operations ")


if __name__ == "__main__":
    root = root_to_symbolize()
    candidates = find_symbol_candidates((root,))
    subtrees = breadth_first_preorder_knode(root)
    print(root)
    print(candidates)

    test_reconstruction()
    test_nested_nodes()
    # display_training_task("2dc579da.json")
    inputs, outputs, input_test, output_test = train_task_to_grids(
        "2dc579da.json"
    )
    input = inputs[2]

    components_by_colors = grid_to_components_by_colors(input)

    # display_components(input)

    cross = max(components_by_colors[frozenset({1})], key=len)
    a_rect = max(components_by_colors[frozenset({3})], key=len)
    rect_with_hole = next(iter(components_by_colors[frozenset({1, 3})]))

    # PointsOperations.print(
    #     coords_to_points(cross), GridOperations.proportions(input)
    # )
    # PointsOperations.print(
    #     coords_to_points(a_rect), GridOperations.proportions(input)
    # )
    # PointsOperations.print(
    #     coords_to_points(rect_with_hole), GridOperations.proportions(input)
    # )

    # st = syntax_tree_at(rect_with_hole, frozenset({1}), Coord(0, 0))
    # print(st)

    # st = syntax_tree_at(cross, frozenset({1}), Coord(0, 5))
    # print(st)
    print("\nCross distribution")
    display_raw_distribution(cross, frozenset({1}))

    print("\nRect distribution")
    display_raw_distribution(a_rect, frozenset({1}))

    print("\nRect with a hole distribution")
    display_raw_distribution(rect_with_hole, frozenset({1}))
