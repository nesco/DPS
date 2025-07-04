"""
Solve ARC-AGI Problems here
"""

# TODO: Solve the mapping issue
# Bug: Distance is not symmetric, I have a 10 between s_22 and s_34 sometimes

import sys
from collections import defaultdict
from collections.abc import Callable, Set
from itertools import combinations
from typing import TypeVar

from arc_syntax_tree import decode_knode
from edit import (
    apply_transformation,
    extended_edit_distance,
)
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import (
    Coord,
    CoordValue,
    CountValue,
    KNode,
    MoveValue,
    NoneValue,
    PaletteValue,
    PrimitiveNode,
    ProductNode,
    RepeatNode,
    RootNode,
    SumNode,
    full_symbolization,
    unsymbolize,
)
from utils.display import display_objects_syntax_trees
from utils.grid import GridOperations, PointsOperations
from utils.loader import train_task_to_grids

sys.setrecursionlimit(10**9)


T = TypeVar("T")
type IndexedElement[T] = tuple[int, T]

def get_pairings(
    sets: list[set[T]],
    distance_tensor: dict[tuple[int, int, T, T], float],
    taken_elements: set[IndexedElement[T]] = set(),
) -> dict[tuple[int, int], list[tuple[T, T]]]:
    pairings: dict[tuple[int, int], list[tuple[T, T]]] = dict()

    print("Distance tensor:")
    for i, j, st1, st2 in distance_tensor:
        print(f"{i}, {j} - {st1} ° {st2} = {distance_tensor[(i, j, st1, st2)]}")

    for i, set1 in enumerate(sets):
        for j, set2 in enumerate(sets):
            # Step 1: Compute the minimum distance for each elements
            if i >= j:
                continue
            min_to_1: dict[T, set[T]] = dict()
            min_to_2: dict[T, set[T]] = dict()

            # Compute the minimum distance for each element in set1
            for a in set1:
                if (i, a) in taken_elements:
                    continue
                mins, min_dist = None, float("inf")
                for b in set2:
                    if (j, b) in taken_elements:
                        continue
                    dist = distance_tensor[(i, j, a, b)]
                    if dist < min_dist:
                        mins, min_dist = set([b]), dist
                    elif dist == min_dist and mins:
                        mins.add(b)
                    elif dist == min_dist and not mins:
                        mins = set([b])
                if mins is not None:
                    min_to_1[a] = mins

            # Compute the minimum distance for each element in set2
            for b in set2:
                if (j, b) in taken_elements:
                    continue
                mins, min_dist = None, float("inf")
                for a in set1:
                    if (i, a) in taken_elements:
                        continue
                    dist = distance_tensor[(i, j, a, b)]
                    if dist < min_dist:
                        mins, min_dist = set([a]), dist
                    elif dist == min_dist and mins:
                        mins.add(a)
                    elif dist == min_dist and not mins:
                        mins = set([a])
                if mins is not None:
                    min_to_2[b] = mins

            # Step 2: Create possible pairings if reciprocity is satisfied
            pairings[i, j] = []
            print("taken: ", taken_elements)
            for a in set1:
                if (i, a) in taken_elements:
                    continue
                for b in set2:
                    if (j, b) in taken_elements:
                        continue
                    if a in min_to_2[b] and b in min_to_1[a]:
                        print(
                            f" pairing ({i},{a!r}) ↔ ({j},{b!r}) distance {distance_tensor[(i, j, a, b)]} == {distance_tensor[(j, i, b, a)]}"
                        )
                        pairings[i, j].append((a, b))

    return pairings


def find_potential_cliques(
    sets, distance_tensor, taken_elements
) -> list[set[IndexedElement[T]]]:
    # Step 1: Compute the pairings
    pairings = get_pairings(sets, distance_tensor, taken_elements)

    # Step 2: Find cliques under transitivity
    cliques: list[tuple[T, ...]] = pairings[0, 1]
    for i in range(2, len(sets)):
        ncliques: list[tuple[T, ...]] = []
        # At step i all the possible cliques of elements from set 0 to set i are formed
        for clique in cliques:
            valid_elements = set(
                element
                for element in sets[i]
                if (i, element) not in taken_elements
            )
            for j in range(i):
                if not valid_elements:  # loop is doomed
                    break
                allowed = {b for a, b in pairings[j, i] if a == clique[j]}
                valid_elements &= allowed
            if not valid_elements:
                continue
            for element in valid_elements:
                ncliques.append(clique + (element,))
        cliques = ncliques

    # Step 3: return potential cliques ordered by total distance
    return [set(enumerate(clique)) for clique in cliques]


def find_cliques(
    sets: list[set[T]], distance: Callable[[T | None, T | None], float]
) -> list[set[IndexedElement[T]]]:
    cliques: list[set[IndexedElement[T]]] = []
    # Step 1: Compute the distance tensor
    distance_tensor: dict[tuple[int, int, T, T], float] = dict()
    for i, set1 in enumerate(sets):
        for j, set2 in enumerate(sets):
            if i == j:
                continue
            for a in set1:
                for b in set2:
                    distance_tensor[i, j, a, b] = distance(a, b)

    taken_elements: set[IndexedElement[T]] = set()
    # Step 2: Compute potential cliques, while there is a potential clique
    # take the one with the lowest total distance and marks it's elements as taken
    # then recompute potential cliques
    while True:
        print("in find_cliques\n\n")
        potential_cliques = find_potential_cliques(
            sets, distance_tensor, taken_elements
        )
        if not potential_cliques:
            break

        # pick the clique of minimal total distance
        clique = min(
            potential_cliques, key=lambda c: total_distance(c, distance)
        )
        cliques.append(clique)

        # mark its members as taken
        taken_elements.update(clique)
        print(f"taken_elements now = {taken_elements}")
        print(f"clique now = {clique}")
    return cliques


def total_distance(
    elements: Set[IndexedElement[T]], distance: Callable[[T, T], float]
) -> float:
    """
    Compute the total pairwise distance for an N-tuple of items.

    Args:
        elements: A set of (index, item) pairs, where index is an int identifier.
        distance: A function taking two items of type T and returning a float distance.

    Returns:
        The sum of distance(x_i, x_j) over all i < j.
    """
    items = list(elements)
    total = 0.0
    for idx1, x1 in items:
        for idx2, x2 in items:
            if idx1 < idx2:
                total += distance(x1, x2)
    return total


def order_cliques(
    cliques: list[set[IndexedElement[T]]], distance: Callable[[T, T], float]
) -> list[set[IndexedElement[T]]]:
    return sorted(cliques, key=lambda clique: total_distance(clique, distance))


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


def problem1(task="2dc579da.json"):
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
        abstracted_sts[-2][-1], abstracted_sts[-1][-1], tuple()
    )
    print(f"d: {d}, op: {op}")
    print(f"Initial: {abstracted_sts[-2][-1]}")
    print(f"Transformed: {apply_transformation(abstracted_sts[-2][-1], op)}")
    print(f"Final: {abstracted_sts[-1][-1]}")

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
    abstracted_sts: list[tuple[KNode[MoveValue], ...]] = []
    offset = 0
    for i, (st_by_go, go_sorted) in enumerate(objects_and_syntax_trees):
        abstracted_sts.append(symbolized[offset : offset + len(go_sorted)])
        n = len(go_sorted)
        print(f"Program n°{i}")

        for st in symbolized[offset : offset + n]:
            print(f"st: {st}")
            unsymbolized = unsymbolize(st, symbol_table)
            display_objects_syntax_trees(
                [unsymbolized], GridOperations.proportions(grids[i])
            )

        offset += n

    d, op = extended_edit_distance(
        abstracted_sts[-2][-1], abstracted_sts[-1][-1], symbol_table
    )

    print("\n--- Finding Stable Cliques ---")
    abstracted_sets = [set(syntax_trees) for syntax_trees in abstracted_sts]

    def distance_f(a: KNode[MoveValue] | None, b: KNode[MoveValue] | None):
        """
        symbolic_distance = extended_edit_distance(a, b, symbol_table)[0]
        shallow_literal_a = a
        shallow_literal_b = b
        compute_literal = False
        if isinstance(a, SymbolNode):
            shallow_literal_a = reduce_abstraction(
                symbol_table[a.index.value], a.parameters
            )
        if isinstance(b, SymbolNode):
            shallow_literal_b = reduce_abstraction(
                symbol_table[b.index.value], b.parameters
            )
        if isinstance(a, SymbolNode) and isinstance(b, SymbolNode):
            return extended_edit_distance(shallow_literal_a, shallow_literal_b, symbol_table)[
                0
            ]
        elif isinstance(a, SymbolNode) or isinstance(b, SymbolNode):
            return float(
                min(
                    extended_edit_distance(a, b)[0],
                    extended_edit_distance(
                        shallow_literal_a, shallow_literal_b
                    )[0],
                )
            )
        """
        # Regularize by the unsymbolized version, should not change anything if both contains no symbols
        c = unsymbolize(a, symbol_table)
        d = unsymbolize(b, symbol_table)

        return float(extended_edit_distance(c, d, symbol_table)[0])

    # Input cliques
    cliques = find_cliques(abstracted_sets[:3], distance_f)

    if cliques:
        print(f"\nFound {len(cliques)} stable clique(s):")
        for i, clique_data in enumerate(
            cliques
        ):  # clique_data is a set of (set_index, element)
            print(f"  Clique {i}:")
            # Sort clique elements by set_idx for consistent printing
            sorted_clique_elements = sorted(
                list(clique_data), key=lambda x: x[0]
            )
            for i, element in enumerate(sorted_clique_elements):
                ind, st = element
                print(f"{ind}: {st}")
                unsymbolized = unsymbolize(st, symbol_table)
                print(unsymbolized)
                for j in range(i):
                    ind2, st2 = sorted_clique_elements[j]
                    dist, ops = extended_edit_distance(
                        unsymbolized,
                        unsymbolize(st2, symbol_table),
                        symbol_table,
                    )
                    print(
                        f"Distance between {ind}: {i, st}={unsymbolized} and {ind2}: {j, st2}={unsymbolize(st2, symbol_table)}: {float(dist)}\n",
                        f"Operations: {ops}",
                    )
                display_objects_syntax_trees(
                    [unsymbolized], GridOperations.proportions(grids[ind])
                )

            # TO - DO

    return symbol_table


def test_distance_symmetrical():
    # Regression issue
    node_6 = SumNode(
        children=frozenset(
            [
                RepeatNode(
                    node=PrimitiveNode(value=MoveValue(0)), count=CountValue(8)
                )
            ]
        )
    )

    node_17 = RootNode(
        node=node_6,
        position=CoordValue(Coord(5, 1)),
        colors=PaletteValue(frozenset({4})),
    )

    node_22 = ProductNode(
        children=(
            node_17,
            RootNode(
                node=NoneValue(None),
                position=CoordValue(Coord(5, 1)),
                colors=PaletteValue(frozenset({1})),
            ),
        )
    )

    node_34 = RootNode(
        node = NoneValue(None),
        position = CoordValue(Coord(1, 2)),
        colors = PaletteValue(frozenset({8})),
    )

    print(node_22, node_34)
    d_1, ops_1 = extended_edit_distance(node_22, node_34, tuple())
    d_2, ops_2 = extended_edit_distance(node_34, node_22, tuple())
    print(f"d_1: {d_1}, ops_1: {ops_1}")
    print(f"d_2: {d_2}, ops_2: {ops_2}")

    # Test 2
    node_19 = RootNode(
        node = SumNode(
            children=frozenset(
                [
                    RepeatNode(
                        node=RepeatNode(node=PrimitiveNode(value=MoveValue(0)), count=CountValue(3)), count=CountValue(4)
                    )
                ]
            )
        ),
        position = CoordValue(Coord(3, 3)),
        colors = PaletteValue(frozenset({2})),
    )

    node_9 = RootNode(
        node = ProductNode(children=(
            PrimitiveNode(value=MoveValue(0)),
            PrimitiveNode(value=MoveValue(6)),
        )),
        position = CoordValue(Coord(1, 3)),
        colors = PaletteValue(frozenset({8})),
    )

    print(node_19, node_9)
    d_1, ops_1 = extended_edit_distance(node_19, node_9, tuple())
    d_2, ops_2 = extended_edit_distance(node_9, node_19, tuple())
    print(f"d_1: {d_1}, ops_1: {ops_1}")
    print(f"d_2: {d_2}, ops_2: {ops_2}")

if __name__ == "__main__":
    sym = problem()
    # test_distance_symmetrical()
