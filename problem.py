"""
Solve ARC-AGI Problems here
"""

from arc_syntax_tree import decode_knode
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import full_symbolization
from utils.grid import GridOperations, PointsOperations
from utils.loader import train_task_to_grids


def problem(task="2dc579da.json"):
    inputs, outputs, input_test, output_test = train_task_to_grids(task)
    grids = inputs + outputs + [input_test]

    objects_and_syntax_trees = [grid_to_syntax_trees(grid) for grid in grids]
    final_st = tuple(
        st_by_go[go_sorted[-1]]
        for st_by_go, go_sorted in objects_and_syntax_trees
    )
    symbolized, symbol_table = full_symbolization(final_st)

    st_by_go, go_sorted = grid_to_syntax_trees(output_test)
    to_guess = st_by_go[go_sorted[-1]]

    for i, s in enumerate(symbol_table):
        print(f"Symbol n°{i}: {s}")

    print("\n\n")

    for i in range(len(inputs)):
        print(f"Grid n°{i}")
        print("Input")
        print(symbolized[i])
        PointsOperations.print(
            decode_knode(final_st[i]), GridOperations.proportions(grids[i])
        )
        print("Output")
        print(symbolized[len(inputs) + i])
        PointsOperations.print(
            decode_knode(final_st[len(inputs) + i]),
            GridOperations.proportions(grids[len(inputs) + i]),
        )

    print(symbolized[-1])
    PointsOperations.print(
        decode_knode(final_st[-1]), GridOperations.proportions(grids[-1])
    )

    print(f"to guess: {to_guess}")
    PointsOperations.print(
        decode_knode(to_guess), GridOperations.proportions(output_test)
    )


if __name__ == "__main__":
    problem()
