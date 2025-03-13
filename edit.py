"""
This module is dedicated to compute edit distance and transformations.
Four valid operations will be used: Identity, Add, Delete, Substitute.
Distances are computed with given distance and "length" functions.
If there is some kind of neutral element in the "set" of objects to compare, the "length" function will usually be the distance to it.

For tree type structures, Identity only mean the current non-tree values of the compared node are similar, as operations on their children will be linearized and thus counted
"""

from dataclasses import dataclass, is_dataclass
from typing import Callable, Generic, Sequence, TypeVar

from localtypes import BitLengthAware
from tree_functionals import subtree_hash

T = TypeVar("T")


@dataclass(frozen=True)
class Identity(Generic[T], BitLengthAware):
    """Represents an element that remains unchanged."""

    value: T

    def bit_length(self) -> int:
        return 0

    def __str__(self):
        return f"Identity: {self.value}"


@dataclass(frozen=True)
class Add(Generic[T], BitLengthAware):
    """Represents an element that is added."""

    value: T

    def bit_length(self) -> int:
        if isinstance(self.value, BitLengthAware):
            return self.value.bit_length()
        return 1

    def __str__(self):
        return f"Add: {self.value}"


@dataclass(frozen=True)
class Delete(Generic[T]):
    """Represents an element that is deleted."""

    value: T

    def bit_length(self) -> int:
        if isinstance(self.value, BitLengthAware):
            return self.value.bit_length()
        return 1

    def __str__(self):
        return f"Delete: {self.value}"


@dataclass(frozen=True)
class Substitute(Generic[T]):
    """Represents an element substituted with another."""

    previous_value: T
    next_value: T

    def bit_length(self) -> int:
        if isinstance(self.previous_value, BitLengthAware) and isinstance(
            self.next_value, BitLengthAware
        ):
            return (
                self.previous_value.bit_length() + self.next_value.bit_length()
            )

        return 2

    def __str__(self):
        return f"Substitute: {self.previous_value} -> {self.next_value}"


type Operation[T] = Identity[T] | Add[T] | Delete[T] | Substitute[T]


# Is it used?
OrderedTransformation = tuple[Operation, int, T | None, T | None]
UnorderedTransformation = tuple[Operation, T | None, T | None]


def sequence_edit_distance(
    list1: Sequence[T],
    list2: Sequence[T],
    distance_func: Callable[[T, T], tuple[int, Sequence[Operation[T]]]],
    len_func: Callable[[T], int],
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sequences using dynamic programming.

    Args:
        list1: First sequence.
        list2: Second sequence.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.

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
            sub_dist, _ = distance_func(list1[i - 1], list2[j - 1])
            substitute_cost = dp[i - 1][j - 1] + sub_dist
            dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

    # Backtrack to find transformations if requested
    transformations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_dist, sub_ops = distance_func(list1[i - 1], list2[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + sub_dist:
                if sub_dist == 0:
                    transformations.append(Identity(list1[i - 1]))
                else:
                    transformations.append(
                        Substitute(list1[i - 1], list2[j - 1])
                    )
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + len_func(list1[i - 1]):
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
    distance_func: Callable[[T, T], tuple[int, Sequence[Operation[T]]]],
    len_func: Callable[[T], int],
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sets using a greedy approach.

    Args:
        set1: First set.
        set2: Second set.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    # Handle empty set cases
    if not set1 and not set2:
        return 0, []
    if not set1:
        dist = sum(len_func(elem) for elem in set2)
        return dist, [Add(elem) for elem in set2]
    if not set2:
        dist = sum(len_func(elem) for elem in set1)
        return dist, [Delete(elem) for elem in set1]

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
                (distance_func(elem1, list2[j])[0], j) for j in available
            )
            if min_dist < len_func(elem1) + len_func(list2[min_j]):
                total_dist += min_dist
                _, ops = distance_func(elem1, list2[min_j])
                if min_dist == 0:
                    transformations.append(Identity(elem1))
                else:
                    transformations.extend(ops)
                available.remove(min_j)
            else:
                total_dist += len_func(elem1)
                transformations.append(Delete(elem1))
        else:
            total_dist += len_func(elem1)
            transformations.append(Delete(elem1))

    # Handle remaining elements in set2
    for j in available:
        total_dist += len_func(list2[j])
        transformations.append(Add(list2[j]))

    return total_dist, tuple(transformations)


U = TypeVar("U", bound=BitLengthAware)


def bitlengthaware_edit_distance(
    a: U, b: U
) -> tuple[int, Sequence[Operation[U]]]:
    """
    Computes the edit distance and transformations between two BitLengthAware objects.

    Args:
        a: First object (assumed to implement bit_length()).
        b: Second object (assumed to implement bit_length()).

    Returns:
        Tuple of (distance: int, transformations: Sequence[Operation[T]]).
    """
    # Check if subtrees are identical
    if subtree_hash(a) == subtree_hash(b):
        return 0, [Identity(a)]

    # Handle non-dataclass or primitive types
    if not is_dataclass(a) or not is_dataclass(b):
        if a == b:
            return 0, [Identity(a)]
        return a.bit_length() + b.bit_length(), [Substitute(a, b)]

    # Different types require substitution
    if not isinstance(a, type(b)):
        return a.bit_length() + b.bit_length(), [Substitute(a, b)]

    # Compare fields of matching dataclass types
    total_distance = 0
    transformations = []
    from dataclasses import fields

    for field in fields(a):
        a_field = getattr(a, field.name)
        b_field = getattr(b, field.name)
        if isinstance(a_field, (tuple, list)):
            dist, ops = sequence_edit_distance(
                a_field,
                b_field,
                distance_func=bitlengthaware_edit_distance,
                len_func=lambda x: x.bit_length(),
            )
            total_distance += dist
            transformations.extend(ops)
        elif isinstance(a_field, set):
            dist, ops = set_edit_distance(
                a_field,
                b_field,
                distance_func=bitlengthaware_edit_distance,
                len_func=lambda x: x.bit_length(),
            )
            total_distance += dist
            transformations.extend(ops)
        else:
            dist, ops = bitlengthaware_edit_distance(a_field, b_field)
            total_distance += dist
            transformations.extend(ops)

    # If no changes, return Identity
    if total_distance == 0:
        return 0, [Identity(a)]
    # Otherwise, return transformations as-is
    else:
        return total_distance, transformations


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
        distance_func=lambda x, y: (0, (Identity(x),))
        if x == y
        else (2, (Substitute(x, y),)),
        len_func=lambda x: 1,
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
    assert distance == sum(op.bit_length() for op in transformations), (
        "The sum of the operations bit length should math the distance"
    )
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
        distance_func=lambda x, y: (0, (Identity(x),))
        if x == y
        else (2, (Substitute(x, y),)),
        len_func=lambda x: 1,
    )
    assert distance == sum(op.bit_length() for op in transformations), (
        "The sum of the operations bit length should math the distance"
    )
    print(f"Distance between {set1} and {set2}: {distance}")
    print("Transformations:")
    for transformation in transformations:
        print(str(transformation))


if __name__ == "__main__":
    test_edit_distance()
    print()  # Add a newline between test outputs
    test_set_edit_distance()
