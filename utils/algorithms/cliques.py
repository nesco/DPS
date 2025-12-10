"""
Clique-finding algorithms for matching objects across sets.

Given multiple sets and a distance function, finds cliques where each
clique contains exactly one element from each set, with elements being
mutually close (k-reciprocal nearest neighbors).

Complexity (s=sets, n=elements/set, k=neighbor count, C=cliques found):
- Distance tensor: O(s² × n² × D) once, where D = distance function cost
- Per greedy iteration: O(s² × n² log k) pairings + O(C × s²) selection
- Total iterations: O(n)
"""

import heapq
import logging
from collections.abc import Callable, Sequence, Set
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

type IndexedElement[T] = tuple[int, T]
type DistanceTensor[T] = dict[tuple[int, int, T, T], float]
type KNearestNeighbors[T] = dict[tuple[int, int], dict[T, set[T]]]
type PairingIndex[T] = dict[tuple[int, int], dict[T, set[T]]]


def _build_distance_tensor(
    sets: Sequence[Set[T]],
    distance: Callable[[T | None, T | None], float],
) -> DistanceTensor[T]:
    """Precompute distances between all element pairs from different sets."""
    tensor: DistanceTensor[T] = {}
    for i, set_i in enumerate(sets):
        for j, set_j in enumerate(sets):
            if i != j:
                for a in set_i:
                    for b in set_j:
                        tensor[i, j, a, b] = distance(a, b)
    return tensor


def _compute_k_nearest(
    sets: Sequence[Set[T]],
    tensor: DistanceTensor[T],
    taken: Set[tuple[int, T]],
    k: int,
) -> KNearestNeighbors[T]:
    """
    For each element a in set i, find its k nearest neighbors in set j.

    Returns:
        knn[(i, j)][a] = {b₁, b₂, ...} where each b is among a's k nearest in set j.
        Ties at the k-th position are included.
    """
    nsets = len(sets)
    knn: KNearestNeighbors[T] = {}

    for i in range(nsets):
        for j in range(nsets):
            if i == j:
                continue

            knn[(i, j)] = {}
            for a in sets[i]:
                if (i, a) in taken:
                    continue

                distances_to_j = [
                    (tensor[i, j, a, b], b) for b in sets[j] if (j, b) not in taken
                ]
                if not distances_to_j:
                    continue

                if k >= len(distances_to_j):
                    k_nearest = distances_to_j
                else:
                    k_nearest = heapq.nsmallest(k, distances_to_j, key=lambda x: x[0])

                # Include ties at the cutoff distance
                cutoff_distance = k_nearest[-1][0]
                knn[(i, j)][a] = {
                    b for dist, b in distances_to_j if dist <= cutoff_distance
                }

    return knn


def _compute_reciprocal_pairs(
    sets: Sequence[Set[T]],
    knn: KNearestNeighbors[T],
    taken: Set[tuple[int, T]],
) -> PairingIndex[T]:
    """
    Find reciprocal k-nearest neighbor pairs.

    A pair (a, b) is reciprocal if b is among a's k-nearest AND a is among b's k-nearest.

    Returns:
        pairings[(i, j)][a] = {b₁, b₂, ...} for i < j, enabling O(1) neighbor lookup.
    """
    nsets = len(sets)
    pairings: PairingIndex[T] = {}

    for i in range(nsets):
        for j in range(i + 1, nsets):
            pairings[(i, j)] = {}

            for a in sets[i]:
                if (i, a) in taken or a not in knn[(i, j)]:
                    continue

                reciprocal_neighbors = {
                    b
                    for b in knn[(i, j)][a]
                    if b in knn[(j, i)] and a in knn[(j, i)][b]
                }

                if reciprocal_neighbors:
                    pairings[(i, j)][a] = reciprocal_neighbors

    return pairings


def _build_pairing_index(
    sets: Sequence[Set[T]],
    tensor: DistanceTensor[T],
    taken: Set[tuple[int, T]],
    k: int = 2,
) -> PairingIndex[T]:
    """Build indexed k-reciprocal nearest neighbor pairings, excluding taken elements."""
    knn = _compute_k_nearest(sets, tensor, taken, k)
    return _compute_reciprocal_pairs(sets, knn, taken)


def _find_potential_cliques(
    pairings: PairingIndex[T],
    nsets: int,
) -> list[tuple[T, ...]]:
    """
    Build all potential cliques by extending pairwise pairings transitively.

    A clique (e₀, e₁, ..., eₙ) is valid iff for all i < j, eᵢ and eⱼ are reciprocal neighbors.
    """
    if nsets < 2:
        return []

    # Start with reciprocal pairs from sets 0 and 1
    cliques: list[tuple[T, ...]] = [
        (a, b) for a, neighbors in pairings.get((0, 1), {}).items() for b in neighbors
    ]

    # Extend each partial clique to include one element from each remaining set
    for set_idx in range(2, nsets):
        extended_cliques: list[tuple[T, ...]] = []

        for clique in cliques:
            # Element must be a reciprocal neighbor of ALL existing clique members
            valid_extensions: set[T] | None = None

            for member_idx in range(set_idx):
                member_element = clique[member_idx]
                neighbors_in_target = pairings.get((member_idx, set_idx), {}).get(
                    member_element, set()
                )

                if valid_extensions is None:
                    valid_extensions = set(neighbors_in_target)
                else:
                    valid_extensions &= neighbors_in_target

                if not valid_extensions:
                    break

            if valid_extensions:
                for ext in valid_extensions:
                    extended_cliques.append(clique + (ext,))

        cliques = extended_cliques

    return cliques


def _clique_total_distance(
    clique: tuple[T, ...],
    tensor: DistanceTensor[T],
) -> float:
    """Sum of pairwise distances between all clique members."""
    total = 0.0
    clique_size = len(clique)
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            total += tensor[i, j, clique[i], clique[j]]
    return total


def find_cliques(
    sets: list[set[T]], distance: Callable[[T | None, T | None], float]
) -> list[set[IndexedElement[T]]]:
    """
    Find cliques of matching elements across multiple sets.

    Uses k-reciprocal nearest neighbors for stability, then greedily
    selects cliques by minimum total pairwise distance.

    Args:
        sets: List of sets, one per grid/domain.
        distance: Distance function between elements.

    Returns:
        List of cliques, where each clique is a set of (set_index, element) pairs.
    """
    if len(sets) < 2:
        return []

    nsets = len(sets)
    tensor = _build_distance_tensor(sets, distance)
    logger.debug(f"Distance tensor: {len(tensor)} entries")

    cliques: list[set[IndexedElement[T]]] = []
    taken: set[IndexedElement[T]] = set()

    while True:
        # Rebuild pairings (k-nearest changes as elements are taken)
        pairings = _build_pairing_index(sets, tensor, taken, k=2)
        candidates = _find_potential_cliques(pairings, nsets)

        if not candidates:
            break

        best = min(candidates, key=lambda c: _clique_total_distance(c, tensor))
        cliques.append(set(enumerate(best)))

        for i, elem in enumerate(best):
            taken.add((i, elem))

        logger.debug(f"Found clique with {len(best)} elements")

    return cliques


def total_distance(
    elements: Set[IndexedElement[T]], distance: Callable[[T, T], float]
) -> float:
    """
    Compute sum of pairwise distances within a clique.

    For internal use when no precomputed tensor is available.
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
    """Sort cliques by total internal distance (tightest first)."""
    return sorted(cliques, key=lambda c: total_distance(c, distance))
