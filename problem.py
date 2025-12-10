"""
Solve ARC-AGI tasks by matching objects across grids.

Finds semantic correspondences between objects in different grids
by clustering them into cliques based on structural similarity.

Distance metrics:
- edit: Raw edit distance (transformation cost)
- nid: Normalized Information Distance (AIT-grounded)
- structural: Normalized structural distance (ignores position/color)
"""

import logging
import sys
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TypedDict

from edit import (
    extended_edit_distance,
    normalized_information_distance,
    structural_distance_value,
)
from hierarchy import grid_to_syntax_trees
from kolmogorov_tree import (
    KNode,
    MoveValue,
    full_symbolization,
    unsymbolize,
)
from arc.types import ColorGrid
from utils.cliques import IndexedElement, find_cliques
from utils.display import display_objects_syntax_trees
from utils.grid import GridOperations
from utils.loader import train_task_to_grids

sys.setrecursionlimit(10**9)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Distance metrics for object matching."""

    EDIT = "edit"
    NID = "nid"
    STRUCTURAL = "structural"


SyntaxTree = KNode[MoveValue]
SymbolTable = Sequence[SyntaxTree]


class TaskResult(TypedDict):
    """Result of solving an ARC task."""

    symbol_table: SymbolTable
    cliques: list[set[IndexedElement[SyntaxTree]]]
    grids: list[ColorGrid]
    syntax_trees_by_grid: list[tuple[SyntaxTree, ...]]


def create_distance_function(
    metric: DistanceMetric,
    symbol_table: SymbolTable,
) -> Callable[[SyntaxTree | None, SyntaxTree | None], float]:
    """Create a distance function for the given metric."""

    def distance(a: SyntaxTree | None, b: SyntaxTree | None) -> float:
        if a is None or b is None:
            return float("inf") if a != b else 0.0

        node_a = unsymbolize(a, symbol_table)
        node_b = unsymbolize(b, symbol_table)

        match metric:
            case DistanceMetric.EDIT:
                return float(extended_edit_distance(node_a, node_b, symbol_table)[0])
            case DistanceMetric.NID:
                return normalized_information_distance(node_a, node_b, symbol_table)
            case DistanceMetric.STRUCTURAL:
                return structural_distance_value(node_a, node_b, symbol_table)

    return distance


def _grids_to_symbolized_syntax_trees(
    grids: list[ColorGrid],
) -> tuple[list[tuple[SyntaxTree, ...]], SymbolTable]:
    """
    Convert grids to syntax trees with shared symbolization.

    Returns syntax trees grouped by grid, plus the shared symbol table.
    """
    grid_decompositions = [grid_to_syntax_trees(grid) for grid in grids]

    all_syntax_trees = tuple(
        syntax_tree
        for trees_by_object, objects in grid_decompositions
        for syntax_tree in (trees_by_object[obj] for obj in objects)
    )

    symbolized_trees, symbol_table = full_symbolization(all_syntax_trees)

    # Partition symbolized trees back into per-grid groups
    trees_by_grid: list[tuple[SyntaxTree, ...]] = []
    offset = 0
    for trees_by_object, objects in grid_decompositions:
        count = len(objects)
        trees_by_grid.append(symbolized_trees[offset : offset + count])
        offset += count

    return trees_by_grid, symbol_table


def _grid_label(grid_index: int, num_inputs: int) -> str:
    """Return 'input', 'output', or 'test' based on grid position."""
    if grid_index < num_inputs:
        return "input"
    elif grid_index < 2 * num_inputs:
        return "output"
    else:
        return "test"


def _log_cliques(
    cliques: list[set[IndexedElement[SyntaxTree]]],
    symbol_table: SymbolTable,
    grids: list[ColorGrid],
    distance: Callable[[SyntaxTree | None, SyntaxTree | None], float],
    show_visuals: bool,
) -> None:
    """Log clique contents and optionally display visuals."""
    for clique_idx, clique in enumerate(cliques):
        members = sorted(clique, key=lambda x: x[0])

        logger.info(f"Clique {clique_idx}:")
        for grid_idx, tree in members:
            unsymbolized = unsymbolize(tree, symbol_table)
            logger.info(f"  Grid {grid_idx}: {tree}")
            logger.debug(f"    Unsymbolized: {unsymbolized}")

            if show_visuals:
                display_objects_syntax_trees(
                    [unsymbolized],  # type: ignore[list-item]
                    GridOperations.proportions(grids[grid_idx]),
                )

        if len(members) > 1:
            logger.debug("  Pairwise distances:")
            for i, (idx1, tree1) in enumerate(members):
                for idx2, tree2 in members[:i]:
                    d = distance(tree1, tree2)
                    logger.debug(f"    ({idx1},{idx2}): {d:.3f}")


def solve_task(
    task: str = "2dc579da.json",
    metric: DistanceMetric = DistanceMetric.STRUCTURAL,
    verbose: bool = True,
    show_visuals: bool = True,
) -> TaskResult:
    """
    Solve an ARC task by finding cliques of matching objects across grids.

    Args:
        task: Task filename to solve.
        metric: Distance metric to use for matching.
        verbose: Whether to log detailed information.
        show_visuals: Whether to display visual representations.

    Returns:
        TaskResult with symbol_table, cliques, grids, and syntax_trees_by_grid.
    """
    logger.info(f"Solving task: {task}, metric: {metric.value}")

    inputs, outputs, input_test, _ = train_task_to_grids(task)
    grids = inputs + outputs + [input_test]
    num_inputs = len(inputs)
    logger.info(f"Loaded {num_inputs} input-output pairs + 1 test input")

    trees_by_grid, symbol_table = _grids_to_symbolized_syntax_trees(grids)

    if verbose:
        logger.info(f"Symbol table: {len(symbol_table)} symbols")
        for i, sym in enumerate(symbol_table):
            logger.debug(f"  s_{i}: {sym}")

        for grid_idx, trees in enumerate(trees_by_grid):
            label = _grid_label(grid_idx, num_inputs)
            logger.info(f"Grid {grid_idx} ({label}): {len(trees)} objects")
            for tree in trees:
                logger.debug(f"  {tree}")

    distance = create_distance_function(metric, symbol_table)

    logger.info(f"Finding cliques across {num_inputs} input grids...")
    input_sets = [set(trees) for trees in trees_by_grid[:num_inputs]]
    cliques = find_cliques(input_sets, distance)

    logger.info(f"Found {len(cliques)} clique(s)")
    if verbose and cliques:
        _log_cliques(cliques, symbol_table, grids, distance, show_visuals)

    return {
        "symbol_table": symbol_table,
        "cliques": cliques,
        "grids": grids,
        "syntax_trees_by_grid": trees_by_grid,
    }


def problem(task: str = "2dc579da.json") -> SymbolTable:
    """Legacy interface - use solve_task() for new code."""
    return solve_task(task, verbose=True, show_visuals=True)["symbol_table"]


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

    solve_task(
        task=args.task,
        metric=metric_map[args.metric],
        verbose=True,
        show_visuals=not args.no_visuals,
    )
