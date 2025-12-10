"""
Solve ARC-AGI Problems here.

This module implements clique-finding for matching objects across grids.
Objects that play the same semantic role should be grouped together.

Distance metrics available:
- edit: Raw edit distance (transformation cost)
- nid: Normalized Information Distance (AIT-grounded)
- structural: Normalized structural distance (ignores position/color)
"""

import logging
import sys
from collections import Counter
from collections.abc import Callable, Set
from enum import Enum
from itertools import combinations
from typing import TypeVar

from arc import decode_knode
from edit import (
    apply_transformation,
    extended_edit_distance,
    normalized_information_distance,
    structural_distance_value,
)
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import (
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Available distance metrics for clique finding."""

    EDIT = "edit"  # Raw edit distance
    NID = "nid"  # Normalized Information Distance
    STRUCTURAL = "structural"  # Structural distance (ignores position/color)


T = TypeVar("T")
type IndexedElement[T] = tuple[int, T]


# Ideas: not only the minimum but the two-minimuma so it blocks less?
def get_pairings_old(
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
                        print(f" pairing ({i},{a!r}) ↔ ({j},{b!r}) ")
                        pairings[i, j].append((a, b))

    return pairings


def get_pairings(
    sets: list[Set[T]],
    distance_tensor: dict[tuple[int, int, T, T], float],
    taken_elements: Set[tuple[int, T]] = set(),
    *,  # ← keyword-only from here on
    k: int = 1,  # k-reciprocal nearest neighbours
) -> dict[tuple[int, int], list[tuple[T, T]]]:
    """
    Return, for every unordered pair of distinct sets (i,j)  with i < j,
    a list of element pairs (a∈sets[i],  b∈sets[j]) such that

        • b is among the k closest *remaining* elements to a in set-j, and
        • a is among the k closest *remaining* elements to b in set-i.

    If k is None or k <= 0 the whole other set is considered
    (“reciprocal *any* neighbour”).
    """
    pairings: dict[tuple[int, int], list[tuple[T, T]]] = {}

    # -----------------------------------------------------------------
    # helper: k-nearest (with tie inclusion) from src_set to dst_set
    # -----------------------------------------------------------------
    def knearest(
        src_id: int, dst_id: int, src_set: Set[T], dst_set: Set[T]
    ) -> dict[T, Set[T]]:
        res: dict[T, Set[T]] = {}
        for a in src_set:
            if (src_id, a) in taken_elements:
                continue

            # build & sort the distance list once
            dlist = [
                (distance_tensor[(src_id, dst_id, a, b)], b)
                for b in dst_set
                if (dst_id, b) not in taken_elements
            ]
            if not dlist:
                continue
            dlist.sort(key=lambda x: x[0])  # ascending by distance

            if k <= 0 or k >= len(dlist):
                cutoff = dlist[-1][0]  # keep them all
            else:
                cutoff = dlist[k - 1][0]  # distance of the k-th element

            res[a] = {b for dist, b in dlist if dist <= cutoff}
        return res

    # -----------------------------------------------------------------
    # main double loop over unordered pairs of the input sets
    # -----------------------------------------------------------------
    nsets = len(sets)
    for i in range(nsets):
        set_i = sets[i]
        for j in range(i + 1, nsets):
            set_j = sets[j]

            # k-NN maps (a → {b₁,b₂,…}) and (b → {a₁,a₂,…})
            nn_i = knearest(i, j, set_i, set_j)
            nn_j = knearest(j, i, set_j, set_i)

            # reciprocal test
            pij: list[tuple[T, T]] = []
            for a, bs in nn_i.items():
                for b in bs:
                    if a in nn_j.get(b, ()):
                        pij.append((a, b))
            pairings[(i, j)] = pij
    return pairings


def find_potential_cliques(
    sets, distance_tensor, taken_elements
) -> list[set[IndexedElement[T]]]:
    # Step 1: Compute the pairings
    pairings = get_pairings(sets, distance_tensor, taken_elements, k=2)

    # Step 2: Find cliques under transitivity
    cliques: list[tuple[T, ...]] = pairings[0, 1]
    for i in range(2, len(sets)):
        ncliques: list[tuple[T, ...]] = []
        # At step i all the possible cliques of elements from set 0 to set i are formed
        for clique in cliques:
            valid_elements = set(
                element for element in sets[i] if (i, element) not in taken_elements
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
    """
    Find cliques of matching objects across multiple sets.

    Uses k-reciprocal nearest neighbors to find stable pairings,
    then greedily selects cliques by minimum total distance.

    Args:
        sets: List of sets, one per grid.
        distance: Distance function between elements.

    Returns:
        List of cliques, where each clique is a set of (grid_index, element) pairs.
    """
    cliques: list[set[IndexedElement[T]]] = []

    # Step 1: Compute the distance tensor
    logger.debug("Computing distance tensor...")
    distance_tensor: dict[tuple[int, int, T, T], float] = dict()
    for i, set1 in enumerate(sets):
        for j, set2 in enumerate(sets):
            if i == j:
                continue
            for a in set1:
                for b in set2:
                    distance_tensor[i, j, a, b] = distance(a, b)

    logger.debug(f"Distance tensor has {len(distance_tensor)} entries")

    # Log some sample distances at debug level
    if logger.isEnabledFor(logging.DEBUG):
        sample_count = 0
        for (i, j, st1, st2), dist in distance_tensor.items():
            if sample_count < 20:  # Only show first 20
                logger.debug(f"  d({i},{j}): {st1} <-> {st2} = {dist:.3f}")
                sample_count += 1

    taken_elements: set[IndexedElement[T]] = set()

    # Step 2: Greedily find cliques
    iteration = 0
    while True:
        potential_cliques = find_potential_cliques(
            sets, distance_tensor, taken_elements
        )
        if not potential_cliques:
            break

        # Pick the clique of minimal total distance
        clique = min(potential_cliques, key=lambda c: total_distance(c, distance))
        cliques.append(clique)

        # Mark its members as taken
        taken_elements.update(clique)

        iteration += 1
        logger.debug(f"Iteration {iteration}: found clique with {len(clique)} elements")

    return cliques


def find_cliques1(
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

    # print("Distance tensor:")
    # for i, j, st1, st2 in distance_tensor:
    #     print(f"{i}, {j} - {st1} ° {st2} = {distance_tensor[(i, j, st1, st2)]}")

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

        # -------- 1️⃣  original criterion: minimal total distance
        min_dist = min(total_distance(c, distance) for c in potential_cliques)
        best_by_dist = [
            c for c in potential_cliques if total_distance(c, distance) == min_dist
        ]
        best_by_dist = potential_cliques

        # -------- 2️⃣  tie-breaker: edge-rarity (variant #3)
        if len(best_by_dist) == 1:
            clique = best_by_dist[0]
        else:
            pair_freq = Counter()
            for C in best_by_dist:
                for u, v in combinations(C, 2):
                    pair_freq[(u, v)] += 1

            def rarity_score(C, *, eps=1e-9):
                # smaller is better
                base = sum(1.0 / pair_freq[(u, v)] for u, v in combinations(C, 2))
                return base + random.random() * eps  # tiny ε to break exact ties

            print(f"\nsorted cliques = {sorted(best_by_dist, key=rarity_score)}\n")
            clique = min(best_by_dist, key=rarity_score)

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
        st_by_go[go_sorted[-1]] for st_by_go, go_sorted in objects_and_syntax_trees
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


def create_distance_function(
    metric: DistanceMetric,
    symbol_table: tuple,
) -> Callable[[KNode[MoveValue] | None, KNode[MoveValue] | None], float]:
    """
    Create a distance function based on the selected metric.

    Args:
        metric: Which distance metric to use.
        symbol_table: Symbol table for resolving references.

    Returns:
        Distance function that takes two KNodes and returns a float.
    """

    def distance_f(a: KNode[MoveValue] | None, b: KNode[MoveValue] | None) -> float:
        # Always unsymbolize first for consistent comparison
        node_a = unsymbolize(a, symbol_table)
        node_b = unsymbolize(b, symbol_table)

        match metric:
            case DistanceMetric.EDIT:
                return float(extended_edit_distance(node_a, node_b, symbol_table)[0])
            case DistanceMetric.NID:
                return normalized_information_distance(node_a, node_b, symbol_table)
            case DistanceMetric.STRUCTURAL:
                return structural_distance_value(node_a, node_b, symbol_table)

    return distance_f


def solve_task(
    task: str = "2dc579da.json",
    metric: DistanceMetric = DistanceMetric.STRUCTURAL,
    verbose: bool = True,
    show_visuals: bool = True,
) -> dict:
    """
    Solve an ARC task by finding cliques of matching objects across grids.

    Args:
        task: Task filename to solve.
        metric: Distance metric to use for matching.
        verbose: Whether to log detailed information.
        show_visuals: Whether to display visual representations.

    Returns:
        Dictionary containing:
        - symbol_table: The shared symbol table
        - cliques: Found cliques of matching objects
        - grids: The input/output grids
        - syntax_trees: Syntax trees per grid
    """
    logger.info(f"Solving task: {task}")
    logger.info(f"Distance metric: {metric.value}")

    # Load grids
    inputs, outputs, input_test, output_test = train_task_to_grids(task)
    grids = inputs + outputs + [input_test]
    logger.info(f"Loaded {len(inputs)} input-output pairs + 1 test input")

    # Convert grids to syntax trees
    objects_and_syntax_trees = [grid_to_syntax_trees(grid) for grid in grids]

    # Combine all syntax trees for co-symbolization
    all_sts = tuple(
        st_by_go[go]
        for st_by_go, go_sorted in objects_and_syntax_trees
        for go in go_sorted
    )

    symbolized, symbol_table = full_symbolization(all_sts)

    # Log symbol table
    if verbose:
        logger.info(f"Symbol table ({len(symbol_table)} symbols):")
        for i, sym in enumerate(symbol_table):
            logger.debug(f"  s_{i}: {sym}")

    # Reconstruct syntax trees per grid
    abstracted_sts: list[tuple[KNode[MoveValue], ...]] = []
    offset = 0
    for i, (st_by_go, go_sorted) in enumerate(objects_and_syntax_trees):
        grid_sts = symbolized[offset : offset + len(go_sorted)]
        abstracted_sts.append(grid_sts)

        if verbose:
            grid_type = (
                "input"
                if i < len(inputs)
                else ("output" if i < 2 * len(inputs) else "test")
            )
            logger.info(f"Grid {i} ({grid_type}): {len(grid_sts)} objects")
            for st in grid_sts:
                logger.debug(f"  {st}")

        offset += len(go_sorted)

    # Create distance function
    distance_f = create_distance_function(metric, symbol_table)

    # Find cliques in input grids only (first N grids where N = len(inputs))
    logger.info(f"Finding cliques across {len(inputs)} input grids...")
    abstracted_sets = [set(syntax_trees) for syntax_trees in abstracted_sts]
    cliques = find_cliques(abstracted_sets[: len(inputs)], distance_f)

    # Log and display cliques
    logger.info(f"Found {len(cliques)} clique(s)")

    if verbose and cliques:
        for clique_idx, clique_data in enumerate(cliques):
            sorted_elements = sorted(list(clique_data), key=lambda x: x[0])

            logger.info(f"Clique {clique_idx}:")
            for grid_idx, st in sorted_elements:
                unsym = unsymbolize(st, symbol_table)
                logger.info(f"  Grid {grid_idx}: {st}")
                logger.debug(f"    Unsymbolized: {unsym}")

                if show_visuals:
                    display_objects_syntax_trees(
                        [unsym], GridOperations.proportions(grids[grid_idx])
                    )

            # Log pairwise distances within clique
            if len(sorted_elements) > 1:
                logger.debug("  Pairwise distances:")
                for i, (idx1, st1) in enumerate(sorted_elements):
                    for idx2, st2 in sorted_elements[:i]:
                        d = distance_f(st1, st2)
                        logger.debug(f"    ({idx1},{idx2}): {d:.3f}")

    return {
        "symbol_table": symbol_table,
        "cliques": cliques,
        "grids": grids,
        "syntax_trees": abstracted_sts,
    }


# Backward compatible alias
def problem(task="2dc579da.json"):
    """Legacy interface - use solve_task() for new code."""
    return solve_task(task, verbose=True, show_visuals=True)["symbol_table"]


def test_distance_symmetrical():
    # Regression issue
    node_6 = SumNode(
        children=frozenset(
            [RepeatNode(node=PrimitiveNode(value=MoveValue(0)), count=CountValue(8))]
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
        node=NoneValue(None),
        position=CoordValue(Coord(1, 2)),
        colors=PaletteValue(frozenset({8})),
    )

    print(node_22, node_34)
    d_1, ops_1 = extended_edit_distance(node_22, node_34, tuple())
    d_2, ops_2 = extended_edit_distance(node_34, node_22, tuple())
    print(f"d_1: {d_1}, ops_1: {ops_1}")
    print(f"d_2: {d_2}, ops_2: {ops_2}")

    # Test 2
    node_19 = RootNode(
        node=SumNode(
            children=frozenset(
                [
                    RepeatNode(
                        node=RepeatNode(
                            node=PrimitiveNode(value=MoveValue(0)), count=CountValue(3)
                        ),
                        count=CountValue(4),
                    )
                ]
            )
        ),
        position=CoordValue(Coord(3, 3)),
        colors=PaletteValue(frozenset({2})),
    )

    node_9 = RootNode(
        node=ProductNode(
            children=(
                PrimitiveNode(value=MoveValue(0)),
                PrimitiveNode(value=MoveValue(6)),
            )
        ),
        position=CoordValue(Coord(1, 3)),
        colors=PaletteValue(frozenset({8})),
    )

    print(node_19, node_9)
    d_1, ops_1 = extended_edit_distance(node_19, node_9, tuple())
    d_2, ops_2 = extended_edit_distance(node_9, node_19, tuple())
    print(f"d_1: {d_1}, ops_1: {ops_1}")
    print(f"d_2: {d_2}, ops_2: {ops_2}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve ARC tasks")
    parser.add_argument("--task", default="2dc579da.json", help="Task filename")
    parser.add_argument(
        "--metric",
        choices=["edit", "nid", "structural"],
        default="structural",
        help="Distance metric to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-visuals", action="store_true", help="Disable visual output"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    metric_map = {
        "edit": DistanceMetric.EDIT,
        "nid": DistanceMetric.NID,
        "structural": DistanceMetric.STRUCTURAL,
    }

    result = solve_task(
        task=args.task,
        metric=metric_map[args.metric],
        verbose=True,
        show_visuals=not args.no_visuals,
    )
