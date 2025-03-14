"""
This module is dedicated to compute edit distance and transformations.
Four valid operations will be used: Identity, Add, Delete, Substitute.
Distances are computed with given distance and "length" functions.
If there is some kind of neutral element in the "set" of objects to compare, the "length" function will usually be the distance to it.

For tree type structures, Identity only mean the current non-tree values of the compared node are similar, as operations on their children will be linearized and thus counted
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from typing import Any, Callable, Generic, Sequence, TypeVar

from dag_functionals import topological_sort
from localtypes import BitLengthAware, Primitive, ensure_all_instances
from tree_functionals import RoseNode, build_hash_hierarchy, cached_hash

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

# Helpers


@cache
def sequence_edit_distance(
    source_sequence: Sequence[T],
    target_sequence: Sequence[T],
    distance_func: Callable[[T, T], tuple[int, Sequence[Operation[T]]]],
    len_func: Callable[[T], int],
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sequences using dynamic programming.

    Args:
        source_sequence: First sequence.
        target_sequence: Second sequence.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    m, n = len(source_sequence), len(target_sequence)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column with cumulative deletion/insertion costs
    for i in range(m + 1):
        dp[i][0] = sum(len_func(source_sequence[k]) for k in range(i))
    for j in range(n + 1):
        dp[0][j] = sum(len_func(target_sequence[k]) for k in range(j))

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            delete_cost = dp[i - 1][j] + len_func(source_sequence[i - 1])
            insert_cost = dp[i][j - 1] + len_func(target_sequence[j - 1])
            sub_dist, _ = distance_func(
                source_sequence[i - 1], target_sequence[j - 1]
            )
            substitute_cost = dp[i - 1][j - 1] + sub_dist
            dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

    # Backtrack to find transformations if requested
    transformations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_dist, sub_ops = distance_func(
                source_sequence[i - 1], target_sequence[j - 1]
            )
            if dp[i][j] == dp[i - 1][j - 1] + sub_dist:
                if sub_dist == 0:
                    transformations.append(Identity(source_sequence[i - 1]))
                else:
                    transformations.append(
                        Substitute(
                            source_sequence[i - 1], target_sequence[j - 1]
                        )
                    )
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + len_func(
            source_sequence[i - 1]
        ):
            transformations.append(Delete(source_sequence[i - 1]))
            i -= 1
        else:
            transformations.append(Add(target_sequence[j - 1]))
            j -= 1
    transformations.reverse()  # Reverse to get transformations in forward order

    return dp[m][n], tuple(transformations)


def set_edit_distance(
    source_set: set[T] | frozenset[T],
    target_set: set[T] | frozenset[T],
    distance_func: Callable[[T, T], tuple[int, Sequence[Operation[T]]]],
    len_func: Callable[[T], int],
) -> tuple[int, Sequence[Operation[T]]]:
    """
    Computes the edit distance between two sets using a greedy approach.

    Args:
        source_set: First set.
        target_set: Second set.
        distance_func: Returns (distance, transformations) between two elements.
        len_func: Function to compute the "length" or cost of an element.

    Returns:
        Tuple of (edit distance, list of transformations).
    """
    # Handle empty set cases
    if not source_set and not target_set:
        return 0, []
    if not source_set:
        dist = sum(len_func(elem) for elem in target_set)
        return dist, [Add(elem) for elem in target_set]
    if not target_set:
        dist = sum(len_func(elem) for elem in source_set)
        return dist, [Delete(elem) for elem in source_set]

    # Convert sets to lists for processing
    source_sequence = list(source_set)
    target_sequence = list(target_set)
    available = set(range(len(target_sequence)))
    total_dist = 0
    transformations = []

    # Greedy matching
    for elem1 in source_sequence:
        if available:
            min_dist, min_j = min(
                (distance_func(elem1, target_sequence[j])[0], j)
                for j in available
            )
            if min_dist < len_func(elem1) + len_func(target_sequence[min_j]):
                total_dist += min_dist
                _, ops = distance_func(elem1, target_sequence[min_j])
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

    # Handle remaining elements in target_set
    for j in available:
        total_dist += len_func(target_sequence[j])
        transformations.append(Add(target_sequence[j]))

    return total_dist, tuple(transformations)


U = TypeVar("U", bound=BitLengthAware)


@dataclass(frozen=True)
class Trim(Generic[U]):
    """When only a children element is kept."""

    parent: U
    children: U

    def bit_length(self) -> int:
        return self.parent.bit_length() - self.children.bit_length()


@dataclass(frozen=True)
class Attach(Generic[U]):
    """When a children is added to a parent."""

    parent: U
    children: U

    def bit_length(self) -> int:
        return self.parent.bit_length() - self.children.bit_length()


type ExtendedOperation[U] = (
    Identity[U] | Add[U] | Delete[U] | Substitute[U] | Trim[U] | Attach[U]
)


def collect_links(
    bla: BitLengthAware,
    hash_to_object: dict[int, BitLengthAware],
    parent_to_children: defaultdict[int, set[int]],
    parent_field: dict[int, str],
):
    # Preorder collection of the relevant hashes
    bla_hash = cached_hash(bla)
    hash_to_object[bla_hash] = bla

    # Depth first propagation of the collection
    if is_dataclass(bla):
        for field in fields(bla):
            attr = getattr(bla, field.name)
            if isinstance(attr, BitLengthAware):
                field_hash = cached_hash(attr)
                parent_to_children[bla_hash].add(field_hash)
                parent_field[field_hash] = field.name
                collect_links(
                    attr, hash_to_object, parent_to_children, parent_field
                )
            if isinstance(attr, tuple):
                attr = ensure_all_instances(attr, BitLengthAware)
                for elem in attr:
                    field_hash = cached_hash(attr)
                    parent_field[field_hash] = field.name
                    collect_links(
                        elem,
                        hash_to_object,
                        parent_to_children,
                        parent_field,
                    )
            if isinstance(attr, frozenset):
                attr = ensure_all_instances(attr, BitLengthAware)
                for elem in attr:
                    field_hash = cached_hash(attr)
                    parent_field[field_hash] = field.name
                    collect_links(
                        elem,
                        hash_to_object,
                        parent_to_children,
                        parent_field,
                    )


def naive_edit_distance_bla_deprecated(
    source: BitLengthAware | None, target: BitLengthAware | None
) -> tuple[int, tuple[Operation[BitLengthAware]]]:
    if source is None and target is None:
        return 0, tuple()  # Maybe Identity(tuple())?

    if source is None:
        return (target.bit_length(), (Add(target),))

    if target is None:
        return (source.bit_length(), (Delete(source),))

    hash_source = cached_hash(source)
    hash_target = cached_hash(target)

    identity = 0, (Identity(source),)
    substitution = (
        source.bit_length() + target.bit_length(),
        (Substitute(source, target),),
    )

    # Object equality
    if hash_source == hash_target:
        return identity

    # One of them is not a dataclass: -> full substitution
    if not (
        is_dataclass(source)
        and is_dataclass(target)
        or isinstance(source, Primitive)
        or isinstance(target, Primitive)
    ):
        return substitution

    # Both of them are dataclass: now it's field-by-field
    #
    total_distance = 0
    transformations = []

    # Sort the fields by name to guarantee the order of operations
    for field in sorted(
        set(fields(source)) | set(fields(target)), key=lambda x: x.name
    ):
        # Hypothesis: a field with the same name will have be either tuple, frozenset or bitlengthaware objects simultaneously
        source_field = getattr(source, field.name)
        target_field = getattr(target, field.name)

        if isinstance(source_field, tuple):
            if not isinstance(target_field, tuple):
                raise TypeError(
                    f"{source_field} is a tuple, but not {target_field}"
                )
            dist, ops = sequence_edit_distance(
                source_field,
                target_field,
                distance_func=recursive_edit_distance,
                len_func=lambda x: x.bit_length(),
            )
        elif isinstance(source_field, frozenset):
            if not isinstance(target_field, frozenset):
                raise TypeError(
                    f"{source_field} is a frozenset, but not {target_field}"
                )
            dist, ops = set_edit_distance(
                source_field,
                target_field,
                distance_func=recursive_edit_distance,
                len_func=lambda x: x.bit_length(),
            )
        elif isinstance(source_field, BitLengthAware) or source is None:
            if not isinstance(target_field, BitLengthAware) or target is None:
                raise TypeError(
                    f"{target_field} is a BitLengthAware, but not {target_field}"
                )
            dist, ops = recursive_edit_distance(source_field, target_field)
        else:
            raise TypeError(f"Unknown type: {source_field} and {target_field}")

        total_distance += dist
        transformations.extend(ops)
    result = (total_distance, tuple(transformations))

    return result


def edit_distance_bla(
    source: BitLengthAware | None, target: BitLengthAware | None
):
    if source is None and target is None:
        return 0, tuple()  # Maybe Identity(tuple())?

    if source is None:
        return (target.bit_length(), (Add(target),))

    if target is None:
        return (source.bit_length(), (Delete(source),))

    # First collect the object topological structure and build the hash -> object map
    hash_to_object: dict[int, BitLengthAware] = {}
    parent_to_children_source: defaultdict[int, set[int]] = defaultdict(set)
    parent_to_children_target: defaultdict[int, set[int]] = defaultdict(set)
    parent_field: dict[int, str] = {}

    collect_links(
        source, hash_to_object, parent_to_children_source, parent_field
    )
    collect_links(
        target, hash_to_object, parent_to_children_target, parent_field
    )

    def helper(
        source_node: int, target_node: int
    ) -> tuple[int, tuple[ExtendedOperation[BitLengthAware], ...]]:
        source_obj = hash_to_object[source_node]
        target_obj = hash_to_object[target_node]

        min_distance, min_transformations = recursive_edit_distance(
            source_obj, target_obj
        )

        # Comparing the target to a child of the source node
        for child in parent_to_children_source[source_node]:
            # The source_obj is Trimmed
            operation = Trim(source_obj, hash_to_object[child])
            # Computing the distance to the child
            child_distance, child_transformations = helper(child, target_node)
            # Trimming then transforming
            transformations = (operation,) + child_transformations
            distance = operation.bit_length() + child_distance
            if distance < min_distance:
                min_distance, min_transformations = (
                    distance,
                    transformations,
                )
        for child in parent_to_children_target[target_node]:
            # The child is Attached to the parent
            operation = Attach(target_obj, hash_to_object[child])
            # Computing the distance to the child
            child_distance, child_transformations = helper(
                source_node, target_node
            )
            # Transforming then attaching
            transformations = child_transformations + (operation,)
            distance = operation.bit_length() + child_distance

            if distance < min_distance:
                min_distance, min_transformations = (
                    distance,
                    transformations,
                )

        return min_distance, min_transformations


def bitlengthaware_edit_distance_deprecated(
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
    if cached_hash(a) == cached_hash(b):
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

    for field in fields(a):
        a_field = getattr(a, field.name)
        b_field = getattr(b, field.name)
        if isinstance(a_field, tuple):
            dist, ops = sequence_edit_distance(
                a_field,
                b_field,
                distance_func=bitlengthaware_edit_distance_deprecated,
                len_func=lambda x: x.bit_length(),
            )
            total_distance += dist
            transformations.extend(ops)
        elif isinstance(a_field, frozenset):
            dist, ops = set_edit_distance(
                a_field,
                b_field,
                distance_func=bitlengthaware_edit_distance_deprecated,
                len_func=lambda x: x.bit_length(),
            )
            total_distance += dist
            transformations.extend(ops)
        else:
            dist, ops = bitlengthaware_edit_distance_deprecated(
                a_field, b_field
            )
            total_distance += dist
            transformations.extend(ops)

    # If no changes, return Identity
    if total_distance == 0:
        return 0, [Identity(a)]
    # Otherwise, return transformations as-is
    else:
        return total_distance, transformations


def bit_length_edit_distance(
    a: BitLengthAware, b: BitLengthAware
) -> tuple[int, Sequence[Operation[BitLengthAware]]]:
    """
    Compute the edit distance between two trees using bottom-up DP in topological order.

    Args:
        a: First tree (root object implementing bit_length()).
        b: Second tree (root object implementing bit_length()).

    Returns:
        Tuple of (distance: int, transformations: Sequence[Operation]).
    """
    # Build hash hierarchies
    hierarchy_a = build_hash_hierarchy(a)
    hierarchy_b = build_hash_hierarchy(b)

    # Collect all unique subtrees
    hash_to_obj: dict[int, BitLengthAware] = {}

    def collect(obj: BitLengthAware, hierarchy):
        stack = [obj]
        seen = set()
        while stack:
            current = stack.pop()
            h = cached_hash(current)
            if h not in hash_to_obj:
                hash_to_obj[h] = current
            if id(current) not in seen and h in hierarchy:
                seen.add(id(current))
                if is_dataclass(current):
                    for f in fields(current):
                        value = getattr(current, f.name)
                        if isinstance(value, (tuple, frozenset)):
                            stack.extend(value)
                        else:
                            stack.append(value)
                elif isinstance(current, (tuple, frozenset)):
                    stack.extend(current)

    collect(a, hierarchy_a)
    collect(b, hierarchy_b)

    # All unique hashes
    all_hashes = set(hierarchy_a.keys()) | set(hierarchy_b.keys())

    # Child-to-parent mapping for topological order
    child_to_parents: defaultdict[int, set[int]] = defaultdict(set)
    for h, children in {**hierarchy_a, **hierarchy_b}.items():
        for child_h in children:
            child_to_parents[child_h].add(h)

    # DP table
    dp: dict[
        tuple[int, int], tuple[int, Sequence[Operation[BitLengthAware]]]
    ] = {}

    # Queue for bottom-up processing (leaves first)
    queue = deque([h for h in all_hashes if h not in child_to_parents])
    processed = set(queue)

    while queue:
        ha = queue.popleft()
        for hb in all_hashes:
            if (ha, hb) in dp:
                continue

            obj_a = hash_to_obj[ha]
            obj_b = hash_to_obj[hb]

            if ha == hb:
                dp[(ha, hb)] = (0, [Identity(obj_a)])
                continue

            # Handle non-dataclasses
            if not (is_dataclass(obj_a) and is_dataclass(obj_b)):
                cost = obj_a.bit_length() + obj_b.bit_length()
                dp[(ha, hb)] = (cost, [Substitute(obj_a, obj_b)])
                continue

            # Dataclasses: align fields
            fields_a = {f.name: getattr(obj_a, f.name) for f in fields(obj_a)}
            fields_b = {f.name: getattr(obj_b, f.name) for f in fields(obj_b)}
            all_fields = set(fields_a.keys()) | set(fields_b.keys())

            total_distance = 0
            transformations = []

            for fname in all_fields:
                if fname not in fields_a:
                    value_b = fields_b[fname]
                    cost = compute_bit_length(value_b)
                    total_distance += cost
                    transformations.append(Add(value_b))
                elif fname not in fields_b:
                    value_a = fields_a[fname]
                    cost = compute_bit_length(value_a)
                    total_distance += cost
                    transformations.append(Delete(value_a))
                else:
                    ha_field = cached_hash(fields_a[fname])
                    hb_field = cached_hash(fields_b[fname])
                    if (ha_field, hb_field) not in dp:
                        continue  # Children not processed yet
                    dist, ops = dp[(ha_field, hb_field)]
                    total_distance += dist
                    transformations.extend(ops)

            dp[(ha, hb)] = (total_distance, transformations)
            print(
                f"distance between {hash_to_obj[ha]} and {hash_to_obj[hb]} := {dp[(ha, hb)]}"
            )

        # Add parents whose children are all processed
        if ha in child_to_parents:
            for parent_h in child_to_parents[ha]:
                children = hierarchy_a.get(parent_h, set()) | hierarchy_b.get(
                    parent_h, set()
                )
                if (
                    all(child_h in processed for child_h in children)
                    and parent_h not in processed
                ):
                    queue.append(parent_h)
                    processed.add(parent_h)

    # Return distance between roots
    ha_root = cached_hash(a)
    hb_root = cached_hash(b)
    return dp.get((ha_root, hb_root), (0, []))


@cache
def recursive_edit_distance(
    a: BitLengthAware, b: BitLengthAware
) -> tuple[int, tuple[Operation[BitLengthAware], ...]]:
    dp = {}

    def helper(
        a: BitLengthAware, b: BitLengthAware
    ) -> tuple[int, Sequence[Operation[BitLengthAware]]]:
        ha = cached_hash(a)
        hb = cached_hash(b)
        if (ha, hb) in dp:
            return dp[(ha, hb)]

        if ha == hb:
            result = (0, [Identity(a)])
        elif (
            not (is_dataclass(a) and is_dataclass(b))
            or isinstance(a, Primitive)
            or isinstance(b, Primitive)
        ):
            cost = compute_bit_length(a) + compute_bit_length(b)
            result = (cost, [Substitute(a, b)])
        else:
            total_distance = 0
            transformations = []
            for field in sorted(
                set(fields(a)) | set(fields(b)), key=lambda x: x.name
            ):
                a_field = getattr(a, field.name)
                b_field = getattr(b, field.name)
                if isinstance(a_field, tuple):
                    if not isinstance(b_field, tuple):
                        raise TypeError(
                            f"{a_field} is a tuple, but not {b_field}"
                        )
                    dist, ops = sequence_edit_distance(
                        a_field,
                        b_field,
                        distance_func=helper,
                        len_func=compute_bit_length,
                    )
                elif isinstance(a_field, frozenset):
                    if not isinstance(b_field, frozenset):
                        raise TypeError(
                            f"{a_field} is a frozenset, but not {b_field}"
                        )
                    dist, ops = set_edit_distance(
                        a_field,
                        b_field,
                        distance_func=helper,
                        len_func=compute_bit_length,
                    )
                elif isinstance(a_field, BitLengthAware) or a is None:
                    if not isinstance(b_field, BitLengthAware) or b is None:
                        raise TypeError(
                            f"{a_field} is a BitLengthAware, but not {b_field}"
                        )
                    dist, ops = helper(a_field, b_field)
                else:
                    raise TypeError(f"Unknown type: {a_field} and {b_field}")
                total_distance += dist
                transformations.extend(ops)
            result = (total_distance, transformations)

        dp[(ha, hb)] = result
        return result

    distance, transformations = helper(a, b)
    return distance, tuple(transformations)


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
    source_set = frozenset({"a", "b"})
    target_set = frozenset({"a", "c"})
    distance, transformations = set_edit_distance(
        source_set,
        target_set,
        distance_func=lambda x, y: (0, (Identity(x),))
        if x == y
        else (2, (Substitute(x, y),)),
        len_func=lambda x: 1,
    )
    assert distance == sum(op.bit_length() for op in transformations), (
        "The sum of the operations bit length should math the distance"
    )
    print(f"Distance between {source_set} and {target_set}: {distance}")
    print("Transformations:")
    for transformation in transformations:
        print(str(transformation))


def test_bit_length_edit_distance():
    @dataclass(frozen=True)
    class MockValue(Primitive):
        value: int

        def bit_length(self) -> int:
            return 1

    @dataclass(frozen=True)
    class TreeNode(BitLengthAware, RoseNode[MockValue]):
        children: "tuple[TreeNode, ...]" = field(default_factory=tuple)

        def bit_length(self) -> int:
            """Return the size of the subtree (number of nodes)."""
            return self.value.bit_length() + sum(
                child.bit_length() for child in self.children
            )

        def __str__(self):
            return f"{self.value} -> ({'|'.join(str(child) for child in self.children)})"

    # Create primitive values
    prim1 = MockValue(1)
    prim2 = MockValue(2)
    prim3 = MockValue(3)
    prim4 = MockValue(4)
    prim5 = MockValue(5)

    # Create leaf nodes
    leaf1 = TreeNode(prim1)
    leaf2 = TreeNode(prim2)
    leaf3 = TreeNode(prim3)
    leaf4 = TreeNode(prim4)

    # Create test trees
    root1 = TreeNode(prim1, (leaf1, leaf2))  # Tree: 1 -> (1, 2)
    root2 = TreeNode(
        prim1, (leaf1, leaf2)
    )  # Tree: 1 -> (1, 2), identical to root1
    root3 = TreeNode(prim1, (leaf1, leaf3))  # Tree: 1 -> (1, 3)
    root4 = TreeNode(prim1, (leaf1, leaf2, leaf4))  # Tree: 1 -> (1, 2, 4)
    root5 = TreeNode(prim5)  # Tree: 5

    # **Test Case 1: Identical Trees**
    # distance, transformations = recursive_edit_distance(root1, root2)
    distance, transformations = recursive_edit_distance(root1, root2)
    assert distance == 0, "Distance should be 0 for identical trees"
    assert len(transformations) == 1, "Should have one Identity transformation"
    assert isinstance(transformations[0], Identity), (
        "Transformation should be Identity"
    )
    print("Test 1 (Identical trees): Passed")

    # **Test Case 2: One Leaf Value Changed**
    distance, transformations = recursive_edit_distance(root1, root3)
    print(f"distance: {distance}, transformations: {transformations}")
    assert distance == 2, "Distance should be 2 (substitute leaf2 with leaf3)"
    assert any(isinstance(op, Substitute) for op in transformations), (
        "Should include Substitute"
    )
    print("Test 2 (One leaf changed): Passed")

    # **Test Case 3: Added Node**
    distance, transformations = recursive_edit_distance(root1, root4)
    assert distance == 1, "Distance should be 1 (add leaf4)"
    assert any(isinstance(op, Add) for op in transformations), (
        "Should include Add"
    )
    print("Test 3 (Added node): Passed")

    # **Test Case 4: Completely Different Trees**
    distance, transformations = recursive_edit_distance(root1, root5)
    assert distance == 4, (
        "Distance should be 4 (substitute value, delete two children)"
    )
    assert any(isinstance(op, Substitute) for op in transformations), (
        "Should include Substitute"
    )
    assert any(isinstance(op, Delete) for op in transformations), (
        "Should include Delete"
    )
    print("Test 4 (Different trees): Passed")

    print("\nAll tests completed successfully!")
    print("Sample transformations (root1 to root3):")
    _, sample_transforms = bit_length_edit_distance(root1, root3)
    for op in sample_transforms:
        print(f"  {op}")


if __name__ == "__main__":
    test_edit_distance()
    test_set_edit_distance()
    test_bit_length_edit_distance()
