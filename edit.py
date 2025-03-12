"""
This module is dedicated to compute edit distance and transformations.
Four valid operations will be used: Identity, Add, Delete, Substitute.
Distances are computed with given distance and "length" functions.
If there is some kind of neutral element in the "set" of objects to compare, the "length" function will usually be the distance to it.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Identity(Generic[T]):
    """Represents an element that remains unchanged."""

    value: T

    def __str__(self):
        return f"Identity: {self.value}"


@dataclass(frozen=True)
class Add(Generic[T]):
    """Represents an element that is added."""

    value: T

    def __str__(self):
        return f"Add: {self.value}"


@dataclass(frozen=True)
class Delete(Generic[T]):
    """Represents an element that is deleted."""

    value: T

    def __str__(self):
        return f"Delete: {self.value}"


@dataclass(frozen=True)
class Substitute(Generic[T]):
    """Represents an element substituted with another."""

    previous_value: T
    next_value: T

    def __str__(self):
        return f"Substitute: {self.previous_value} -> {self.next_value}"


type Operation[T] = Identity[T] | Add[T] | Delete[T] | Substitute[T]

OrderedTransformation = tuple[Operation, int, T | None, T | None]
UnorderedTransformation = tuple[Operation, T | None, T | None]


def sequence_edit_distance(
    list1: Sequence[T],
    list2: Sequence[T],
    distance_func: Callable[[T, T], int],
    len_func: Callable[[T], int],
    track_changes: bool = False,
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sequences using dynamic programming.

    Args:
        list1: First sequence.
        list2: Second sequence.
        distance_func: Function to compute distance between two elements.
        len_func: Function to compute the "length" or cost of an element.
        track_changes: If True, returns the list of transformations.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column with cumulative deletion/insertion costs
    for i in range(m + 1):
        dp[i][0] = sum(len_func(list1[k]) for k in range(i))
    for j in range(n + 1):
        dp[0][j] = sum(len_func(list2[k]) for k in range(j))

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            delete_cost = dp[i - 1][j] + len_func(list1[i - 1])
            insert_cost = dp[i][j - 1] + len_func(list2[j - 1])
            substitute_cost = dp[i - 1][j - 1] + distance_func(
                list1[i - 1], list2[j - 1]
            )
            dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

    # Backtrack to find transformations if requested
    transformations = []
    if track_changes:
        i, j = m, n
        while i > 0 or j > 0:
            if (
                i > 0
                and j > 0
                and dp[i][j]
                == dp[i - 1][j - 1] + distance_func(list1[i - 1], list2[j - 1])
            ):
                if distance_func(list1[i - 1], list2[j - 1]) == 0:
                    transformations.append(Identity(list1[i - 1]))
                else:
                    transformations.append(
                        Substitute(list1[i - 1], list2[j - 1])
                    )
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + len_func(list1[i - 1]):
                transformations.append(Delete(list1[i - 1]))
                i -= 1
            else:
                transformations.append(Add(list2[j - 1]))
                j -= 1
        transformations.reverse()  # Reverse to get transformations in forward order

    return dp[m][n], tuple(transformations)


def set_edit_distance(
    set1: set[T],
    set2: set[T],
    distance_func: Callable[[T, T], int],
    len_func: Callable[[T], int],
    track_changes: bool = False,
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sets using a greedy approach.

    Args:
        set1: First set.
        set2: Second set.
        distance_func: Function to compute distance between two elements.
        len_func: Function to compute the "length" or cost of an element.
        track_changes: If True, returns the list of transformations.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    # Handle empty set cases
    if not set1 and not set2:
        return 0, []
    if not set1:
        dist = sum(len_func(elem) for elem in set2)
        transformations = [Add(elem) for elem in set2] if track_changes else []
        return dist, transformations
    if not set2:
        dist = sum(len_func(elem) for elem in set1)
        transformations = (
            [Delete(elem) for elem in set1] if track_changes else []
        )
        return dist, transformations

    # Convert sets to lists for processing
    list1 = list(set1)
    list2 = list(set2)
    available = set(range(len(list2)))
    total_dist = 0
    transformations = []

    # Greedy matching
    for elem1 in list1:
        if available:
            min_dist, min_j = min(
                (distance_func(elem1, list2[j]), j) for j in available
            )
            if min_dist < len_func(elem1) + len_func(list2[min_j]):
                total_dist += min_dist
                if track_changes:
                    if min_dist == 0:
                        transformations.append(Identity(elem1))
                    else:
                        transformations.append(Substitute(elem1, list2[min_j]))
                available.remove(min_j)
            else:
                total_dist += len_func(elem1)
                if track_changes:
                    transformations.append(Delete(elem1))
        else:
            total_dist += len_func(elem1)
            if track_changes:
                transformations.append(Delete(elem1))

    # Handle remaining elements in set2
    for j in available:
        total_dist += len_func(list2[j])
        if track_changes:
            transformations.append(Add(list2[j]))

    return total_dist, tuple(transformations) if track_changes else []


def edit_distance(
    a: str, b: str, reverse: bool = True
) -> tuple[int, Sequence[Operation[str]]]:
    """
    Computes the edit distance between two strings.

    Args:
        a: First string.
        b: Second string.
        reverse: If False, reverses the order of transformations.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    distance, transformations = sequence_edit_distance(
        a,
        b,
        distance_func=lambda x, y: 0 if x == y else 1,
        len_func=lambda x: 1,
        track_changes=True,
    )
    if not reverse:
        tuple(reversed(transformations))
    return distance, transformations


### Test Functions


def test_edit_distance():
    """Tests the edit distance computation for strings."""
    a = "kitten"
    b = "sitting"
    distance, transformations = edit_distance(a, b)
    print(f"Distance between {a} and {b}: {distance}")
    print(f"Steps: {len(transformations)}")
    for transformation in transformations:
        print(str(transformation))


def test_set_edit_distance():
    """Tests the edit distance computation for sets."""
    set1 = {"a", "b"}
    set2 = {"a", "c"}
    distance, transformations = set_edit_distance(
        set1,
        set2,
        distance_func=lambda x, y: 0 if x == y else 1,
        len_func=lambda x: 1,
        track_changes=True,
    )
    print(f"Distance between {set1} and {set2}: {distance}")
    print("Transformations:")
    for transformation in transformations:
        print(str(transformation))


if __name__ == "__main__":
    test_edit_distance()
    print()  # Add a newline between test outputs
    test_set_edit_distance()
