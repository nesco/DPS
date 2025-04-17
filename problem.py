"""
Solve ARC-AGI Problems here
"""

from arc_syntax_tree import decode_knode
from edit import apply_transformation, extended_edit_distance
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import full_symbolization
from utils.grid import GridOperations, PointsOperations
from utils.loader import train_task_to_grids


def top_level_only_problem(task="2dc579da.json"):
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

    for i in range(len(inputs)):
        print(
            f"Grid of size {GridOperations.proportions(grids[i])}: {symbolized[i]} -> Grid of size {GridOperations.proportions(grids[len(inputs) + i])}: {symbolized[len(inputs) + i]}"
        )
    print(
        f"Grid of size {GridOperations.proportions(grids[-1])}: {symbolized[-1]} -> Grid of size ?: ?"
    )
    return symbol_table


def problem(task="2dc579da.json"):
    inputs, outputs, input_test, output_test = train_task_to_grids(task)
    grids = inputs + outputs + [input_test]

    objects_and_syntax_trees = [grid_to_syntax_trees(grid) for grid in grids]

    # Combine all syntax trees to symbolize them together
    all_sts = tuple(
        st_by_go[go]
        for st_by_go, go_sorted in objects_and_syntax_trees
        for go in go_sorted
    )

    symbolized, symbol_table = full_symbolization(all_sts)

    print("Symbol Table:")
    for i, sym in enumerate(symbol_table):
        print(f"st n°{i}: {sym}")
    # Reconstruct each list of components for each grid
    abstracted_sts = []
    offset = 0
    for st_by_go, go_sorted in objects_and_syntax_trees:
        print(f"Offset: {offset}")
        abstracted_sts.append(symbolized[offset : offset + len(go_sorted)])
        offset += len(go_sorted)
        for st in symbolized[offset : offset + len(go_sorted)]:
            print(f"st: {st}")

    d, op = extended_edit_distance(
        abstracted_sts[-2][-1], abstracted_sts[-1][-1]
    )
    print(f"d: {d}, op: {op}")
    print(f"Initial: {abstracted_sts[-2][-1]}")
    print(f"Transformed: {apply_transformation(abstracted_sts[-2][-1], op)}")
    print(f"Final: {abstracted_sts[-1][-1]}")

    return symbol_table


if __name__ == "__main__":
    sym = problem()
