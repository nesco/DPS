from arc_syntax_tree import (
    component_to_distribution,
    component_to_raw_syntax_tree_distribution,
    decode_knode
)
from hierarchy import grid_to_components_by_colors
from kolmogorov_tree import MoveValue, ProductNode, RootNode, SumNode
from localtypes import ColorGrid, Colors, Coords, Proportions
from utils.grid import GridOperations, PointsOperations, coords_to_points
from utils.loader import train_task_to_grids

from collections.abc import Sequence

def display_training_task(name: str = "2dc579da.json", include_output=False):
    inputs, outputs, input_test, output_test = train_task_to_grids(name)

    for i in range(len(inputs)):
        print(f"Input n°{i}")
        GridOperations.print(inputs[i])
        print(f"Output n°{i}")
        GridOperations.print(outputs[i])

        print("\n\n")

    print("Input to map")
    GridOperations.print(input_test)

    if include_output:
        print("Output to be mapped")
        GridOperations.print(output_test)


def display_components(grid: ColorGrid):
    components_by_colors = grid_to_components_by_colors(grid)

    for colors, components in components_by_colors.items():
        print(f"\nComponents of colors: {colors}")
        for j, component in enumerate(components):
            print(f"Component n°{j}")
            PointsOperations.print(
                coords_to_points(component), GridOperations.proportions(grid)
            )


def display_raw_distribution(component: Coords, colors: Colors):
    raw_distribution = component_to_raw_syntax_tree_distribution(
        component, colors
    )

    for st in raw_distribution:
        print(f"\nSyntax tree: {st}, length: {st.bit_length()}")


def display_distribution(component: Coords, colors: Colors):
    distribution, symbol_table = component_to_distribution(component, colors)

    print("Distribution: ")
    for st in distribution:
        print(f"{st}: {st.bit_length()} bits")

    print("\nSymbol Table: ")
    for i, st in enumerate(symbol_table):
        print(f"Symbol n°{i}: {st}")

def display_objects_syntax_trees(objects_st: Sequence[ProductNode[MoveValue] | SumNode[MoveValue] | RootNode[MoveValue]], proportions: Proportions):
    for st in objects_st:
        PointsOperations.print(decode_knode(st), proportions)
