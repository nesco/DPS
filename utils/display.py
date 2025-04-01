from arc_syntax_tree import component_to_raw_syntax_tree_distribution
from hierarchy import grid_to_components_by_colors
from kolmogorov_tree import extract_nested_patterns, symbolize
from localtypes import ColorGrid, Colors, Coords
from utils.grid import GridOperations, PointsOperations, coords_to_points
from utils.loader import train_task_to_grids


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

    symbol_table = []
    new_distribution = []
    for st in raw_distribution:
        print(f"\nSyntax tree: {st}, length: {st.bit_length()}")
        new_st = extract_nested_patterns(symbol_table, st)
        print(f"New Syntax tree: {new_st}, length: {new_st.bit_length()}")
        new_distribution.append(new_st)

    co_symbolization, symbol_table = symbolize(
        tuple(new_distribution), tuple(symbol_table)
    )
    for st in co_symbolization:
        print(f"Symbolized Syntax tree: {st}, length: {st.bit_length()}")

    for i, st in enumerate(symbol_table):
        print(f"Symbol n°{i}: {st}")
