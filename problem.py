"""
Solve ARC-AGI Problems here
"""

import sys
from collections import defaultdict
from collections.abc import Callable, Sequence, Set
from itertools import combinations
from typing import TypeVar

from arc_syntax_tree import decode_knode
from edit import apply_transformation, extended_edit_distance
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import (
    KNode,
    MoveValue,
    SymbolNode,
    full_symbolization,
    reduce_abstraction,
)
from utils.grid import GridOperations, PointsOperations
from utils.loader import train_task_to_grids

sys.setrecursionlimit(10**6)


T = TypeVar("T")
type IndexedElement[T] = tuple[int, T]


def find_cliques2(
    sets: Sequence[Set[T]], distance_func: Callable[[T, T], float]
) -> list[set[IndexedElement[T]]]:
    """
    Finds stable cliques among k sets based on a distance function.

    A stable clique is a set of k elements {e_0, e_1, ..., e_{k-1}},
    where e_i is from sets[i], such that for any pair (e_i, e_j) in the clique:
    1. e_j is the closest element in sets[j] to e_i.
    2. e_i is the closest element in sets[i] to e_j.

    This also satisfies an associativity-like property: if (a from A, b from B, c from C)
    form part of a clique, and a is closest to b (in B) and a is closest to c (in C),
    then b will be closest to c (in C) (and all other pairwise mutual closeness
    conditions within {a,b,c} will hold).

    Args:
        sets: A list of k sets. sets[i] is the i-th set.
        distance_func: A function distance_func(el1, el2) that returns a float
                       representing the distance between el1 and el2.

    Returns:
        A list of cliques. Each clique is a set of tuples (set_index, element).
        Example: [{(0, a1), (1, b1)}, {(0, a2), (1, b2)}]
    """

    num_sets = len(sets)
    if num_sets == 0:
        return []

    if num_sets == 1:
        # For a single set, each element is a clique of its own.
        return [{(0, elem)} for elem in sets[0]]

    # Step 1: Compute all-pairs best choices and their distances.
    # best_choice_from_s_to_t[i][j][elem_from_i] = (best_elem_in_j, dist)
    # Stores the best choice in set j for an element from set i.
    best_choice_from_s_to_t: list[list[dict[T, tuple[T | None, float]]]] = [
        [
            defaultdict(lambda: (None, float("inf"))) for _ in range(num_sets)
        ]  # Corrected defaultdict
        for _ in range(num_sets)
    ]

    for i in range(num_sets):
        if not sets[i]:  # Skip if the source set is empty
            continue

        for elem_i in sets[i]:
            for j in range(num_sets):
                if i == j:
                    continue
                if not sets[j]:  # Skip if the target set is empty
                    best_choice_from_s_to_t[i][j][elem_i] = (None, float("inf"))
                    continue

                min_dist_val = float("inf")
                current_best_elem_j = None

                # Find the element in sets[j] closest to elem_i
                for elem_j in sets[j]:
                    dist = distance_func(elem_i, elem_j)
                    if dist < min_dist_val:
                        min_dist_val = dist
                        current_best_elem_j = elem_j
                    # Optional: handle ties in distance. Currently, first one found is kept.

                if current_best_elem_j is not None:
                    best_choice_from_s_to_t[i][j][elem_i] = (
                        current_best_elem_j,
                        min_dist_val,
                    )
                else:  # Should not happen if sets[j] is not empty
                    best_choice_from_s_to_t[i][j][elem_i] = (None, float("inf"))

    print(best_choice_from_s_to_t)

    found_cliques: list[set[tuple[int, T]]] = []

    # Step 2: Recursive function to find cliques
    # partial_clique_elements stores elements [e_0, e_1, ..., e_{current_set_index-1}]
    def find_cliques_recursive(
        current_set_idx: int, partial_clique_elements: list[T]
    ):
        print(f"Partial clique elements: {partial_clique_elements}")
        if current_set_idx == num_sets:
            # Found a full clique
            clique_representation = set()
            for idx, elem in enumerate(partial_clique_elements):
                clique_representation.add((idx, elem))
            found_cliques.append(clique_representation)
            return

        # Iterate through elements of the current set to extend the clique
        for elem_candidate in sets[current_set_idx]:
            is_compatible = True
            # Check if elem_candidate is mutually closest with all elements already in partial_clique_elements
            for existing_elem_set_idx in range(len(partial_clique_elements)):
                existing_elem = partial_clique_elements[existing_elem_set_idx]

                # Check 1: Is existing_elem the best choice for elem_candidate in sets[existing_elem_set_idx]?
                # best_choice_from_s_to_t[current_set_idx][existing_elem_set_idx][elem_candidate]
                choice_info1 = best_choice_from_s_to_t[current_set_idx][
                    existing_elem_set_idx
                ][elem_candidate]
                if not choice_info1 or choice_info1[0] != existing_elem:
                    is_compatible = False
                    break

                # Check 2: Is elem_candidate the best choice for existing_elem in sets[current_set_idx]?
                # best_choice_from_s_to_t[existing_elem_set_idx][current_set_idx][existing_elem]
                choice_info2 = best_choice_from_s_to_t[existing_elem_set_idx][
                    current_set_idx
                ][existing_elem]
                if not choice_info2 or choice_info2[0] != elem_candidate:
                    is_compatible = False
                    break

            if is_compatible:
                partial_clique_elements.append(elem_candidate)
                find_cliques_recursive(
                    current_set_idx + 1, partial_clique_elements
                )
                partial_clique_elements.pop()  # Backtrack

    find_cliques_recursive(0, [])
    return found_cliques


def find_cliques1(
    sets: list[set[T]], distance: Callable[[T | None, T | None], float]
) -> list[set[IndexedElement[T]]]:
    # Step 1: Try to make associations through pairwise comparisons
    associations = defaultdict(lambda: defaultdict(set))
    for (i, variant_i), (j, variant_j) in combinations(enumerate(sets), 2):
        if not variant_i or not variant_j:
            continue  # No variants to match

        edit_matrix = [
            [(distance(a, b), a, b) for b in variant_j] for a in variant_i
        ]

        while edit_matrix and any(edit_matrix):
            # min_dist, a, b = max(
            #    (item for row in edit_matrix for item in row if item),
            #        key=lambda x: distance(None, x[1]) + distance(None, x[2]) - x[0]
            #    )
            min_dist, a, b = min(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[0],
            )

            # If adding the association reduces the total cost, add it
            if min_dist < distance(None, a) + distance(None, b):
                associations[i][a].add((j, b))
                associations[j][b].add((i, a))

            # Remove processed items
            edit_matrix = [
                [item for item in row if item[1] != a and item[2] != b]
                for row in edit_matrix
                if any(item[1] != a for item in row)
            ]

    # Step 2: Remove associations that violates transitivity
    # You want to only retains cliques / complete subgraphes
    changed = True
    while changed:
        changed = False
        # For every sets, for each of it's elements
        # Go to the elements it's associated with and check
        # their are only associated with elements its associated with
        for i in associations:
            for a in list(associations[i].keys()):
                for j, b in list(associations[i][a]):
                    for k, c in list(associations[j][b]):
                        # (i, a) is not in itself as the identity transition is not saved
                        if (k, c) not in associations[i][a] and (k, c) != (
                            i,
                            a,
                        ):
                            # Transitivity violation found, remove the weakest link
                            links = [
                                (
                                    distance(None, a)
                                    + distance(None, b)
                                    - distance(a, b),
                                    (i, a),
                                    (j, b),
                                ),
                                (
                                    distance(None, b)
                                    + distance(None, c)
                                    - distance(b, c),
                                    (j, b),
                                    (k, c),
                                ),
                                (
                                    distance(None, a)
                                    + distance(None, c)
                                    - distance(a, c),
                                    (i, a),
                                    (k, c),
                                ),
                            ]
                            _, (x, y), (z, w) = min(links, key=lambda x: x[0])
                            associations[x][y].discard((z, w))
                            associations[z][w].discard((x, y))
                            changed = True

                            # If an association becomes empty, remove it
                            if not associations[x][y]:
                                del associations[x][y]
                            if not associations[z][w]:
                                del associations[z][w]

    # Step 3: Cluster the clique elements into equivalence classes

    cliques = []
    processed = set()
    for i in associations:
        for a in associations[i]:
            if (i, a) not in processed:
                clique = {(i, a)}
                to_process = [(i, a)]
                while to_process:
                    current = to_process.pop()
                    for associated in associations[current[0]][current[1]]:
                        if associated not in clique:
                            clique.add(associated)
                            to_process.append(associated)

                # Only keep clusters with one element from each set
                if len(clique) == len(sets) and len(
                    set(i for i, _ in clique)
                ) == len(sets):
                    cliques.append(clique)

                processed.update(clique)

    return cliques


def get_pairings(
    sets: list[set[T]],
    distance_tensor: dict[tuple[int, int, T, T], float],
    taken_elements: set[IndexedElement[T]] = set(),
) -> dict[tuple[int, int], list[tuple[T, T]]]:
    pairings: dict[tuple[int, int], list[tuple[T, T]]] = dict()

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
                    dist = distance_tensor[i, j, a, b]
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
                    dist = distance_tensor[i, j, a, b]
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
            for a in set1:
                if (i, a) in taken_elements:
                    continue
                for b in set2:
                    if (j, b) in taken_elements:
                        continue
                    if a in min_to_2[b] and b in min_to_1[a]:
                        print(f"  ↔ pairing ({i},{a!r}) ↔ ({j},{b!r})")
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
                pairings_j_i = pairings[j, i]
                if not pairings_j_i:
                    valid_elements = set()
                    continue
                allowed = {b for a, b in pairings_j_i if a == clique[j]}
                valid_elements &= allowed
            if not valid_elements:
                break
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
        abstracted_sts[-2][-1], abstracted_sts[-1][-1]
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
    for st_by_go, go_sorted in objects_and_syntax_trees:
        abstracted_sts.append(symbolized[offset : offset + len(go_sorted)])
        offset += len(go_sorted)
        for st in symbolized[offset : offset + len(go_sorted)]:
            print(f"st: {st}")

    d, op = extended_edit_distance(
        abstracted_sts[-2][-1], abstracted_sts[-1][-1], symbol_table
    )

    print("\n--- Finding Stable Cliques ---")
    abstracted_sets = [set(syntax_trees) for syntax_trees in abstracted_sts]

    def distance_f(a: KNode[MoveValue] | None, b: KNode[MoveValue] | None):
        symbolic_distance = extended_edit_distance(a, b, symbol_table)[0]
        shallow_literal_a = a
        shallow_literal_b = b
        compute_literal = False
        """
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
            # Regularize by the unsymbolized version, should not change anything if both contains no symbol
        return float(extended_edit_distance(a, b, symbol_table)[0])

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
            for element in sorted_clique_elements:
                print(element)
            # TO - DO

    return symbol_table


if __name__ == "__main__":
    sym = problem()
