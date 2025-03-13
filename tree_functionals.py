"""
Functionals for tree structures. It would work for graphs, if adapted by using `seen = set()`construct to avoid cycles
"""

from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from typing import Callable, Generic, Iterator, Sequence, TypeVar

T = TypeVar("T")


# Test class for a rose tree
@dataclass(frozen=True)
class RoseNode(Generic[T]):
    value: T
    children: "tuple[RoseNode[T], ...]" = field(default_factory=tuple)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__


# Traversals
def breadth_first_preorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """
    Performs a breadth-first preorder traversal of an object, yielding all instances level by level.

    Args:
        after: The function which returns the children of the current object
        root: The root object to traverse.

    Yields:
        T: Each instance in breadth-first preorder.
    """
    if root is None:
        return iter(())
    queue = deque([root])
    while queue:
        current = queue.popleft()
        yield current
        for child in after(current):
            queue.append(child)


def breadth_first_postorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """
    Performs a breadth-first postorder traversal of an object, yielding children before their parents, level by level.

    Args:
        after: The function which returns the children of the current object
        root: The root object to traverse.
    Yields:
        T: Each instance in breadth-first postorder.
    """
    if root is None:
        return iter(())

    # Track nodes by level
    level_map = {}  # Maps level -> nodes at that level
    parent_map = {}  # Maps node id -> parent node

    # Initialize with root at level 0
    queue = deque([(root, 0)])  # (node, level)
    seen = set([id(root)])

    # Build level map in breadth-first order
    while queue:
        node, level = queue.popleft()

        if level not in level_map:
            level_map[level] = []
        level_map[level].append(node)

        for child in after(node):
            if id(child) not in seen:
                seen.add(id(child))
                queue.append((child, level + 1))
                parent_map[id(child)] = node

    # Process levels in reverse order (deepest first)
    for level in sorted(level_map.keys(), reverse=True):
        for node in level_map[level]:
            yield node


def depth_first_preorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """
    Performs a depth-first preorder traversal of an object, yielding the
    current object before its children.

    Args:
        after: The function which returns the children of the current object
        root: The root object to traverse.

    Yields:
        T: Each instance in depth-first preorder.
    """
    if root is None:
        return iter(())
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        # Push subvalues in reverse to process left-to-right
        children = list(after(current))
        for value in reversed(children):
            stack.append(value)


def depth_first_postorder(
    after: Callable[[T], Iterator[T]], root: T | None
) -> Iterator[T]:
    """
    Performs a depth-first postorder traversal of an object, yielding
    children before the current object.

    Args:
        after: The function which returns the children of the current object
        root: The root object to traverse.

    Yields:
        T: Each instance in depth-first postorder.
    """
    if root is None:
        return iter(())
    stack = [(root, False)]
    while stack:
        current, visited = stack.pop()
        if visited:
            yield current
        else:
            stack.append((current, True))
            children = list(after(current))
            for value in reversed(children):
                stack.append((value, False))


# Mappings


def postorder_map(
    node: T,
    f: Callable[[T], T],
    after: Callable[[T], Iterator[T]],
    reconstuct: Callable[[T, Sequence[T]], T],
) -> T:
    """
    Applies a function to each node in a tree in post-order, reconstructing the tree.

    Args:
        node: The root node of the tree.
        f: Function to transform each node after processing children.
        after: Returns an iterator of a node's children.
        reconstruct: Reconstructs a node with its original data and new children.

    Returns:
        The transformed tree.
    """

    new_children = tuple(
        postorder_map(child, f, after, reconstuct) for child in after(node)
    )
    new_node = reconstuct(node, new_children)
    return f(new_node)


def preorder_map(
    node: T,
    f: Callable[[T], T],
    after: Callable[[T], Iterator[T]],
    reconstructor: Callable[[T, list[T]], T],
) -> T:
    """
    Applies a function to each node in a tree in pre-order, reconstructing the tree.

    Args:
        node: The root node of the tree.
        f: Function to transform each node before processing children.
        children_func: Returns an iterator of a node's children.
        reconstructor: Reconstructs a node with its transformed data and new children.

    Returns:
        The transformed tree.
    """

    transformed_node = f(node)
    new_children = [
        preorder_map(child, f, after, reconstructor)
        for child in after(transformed_node)
    ]
    new_node = reconstructor(transformed_node, new_children)
    return new_node


# Dataclass based tree


def dataclass_subvalues(obj: T) -> Iterator[T]:
    """
    Yields all BitLengthAware subvalues of a dataclass object, assuming it's a dataclass.
    For tuple or list fields, yields each BitLengthAware element.

    Args:
        obj: A dataclass object.

    Yields:
        T: Subvalues that are instances of the same class.
    """
    if not is_dataclass(obj):
        return  # Yield nothing if not a dataclass
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, (tuple, list, set)):
            for elem in value:
                yield elem
        else:
            yield value


@cache
def subtree_hash(obj: object) -> int:
    """Recursively compute a hash for a subtree defined using dataclasses and collections."""
    if not is_dataclass(obj):
        if isinstance(obj, tuple):
            return hash(tuple(subtree_hash(elem) for elem in obj))
        elif isinstance(obj, frozenset):
            return hash(tuple(sorted(subtree_hash(elem) for elem in obj)))
        else:
            return hash(obj)
    else:
        field_hashes = tuple(
            (field.name, subtree_hash(getattr(obj, field.name)))
            for field in fields(obj)
        )
        return hash(
            (
                type(obj).__name__,
                tuple(sorted(field_hashes, key=lambda x: x[0])),
            )
        )


def build_hash_hierarchy(obj: object) -> dict[int, frozenset[int]]:
    """
    Create a map where each key is the hash of a subtree and each value is the set of hashes
    of its immediate children, for a tree made of dataclasses and basic collections.

    Args:
        obj: The root of the tree (a dataclass instance, list, tuple, set, or basic object).

    Returns:
        A dictionary mapping each subtree's hash to the set of hashes of its immediate children.
    """
    hash_map: dict[int, frozenset[int]] = {}

    def traverse(node: object) -> None:
        """Recursively traverse the tree, populating hash_map with subtree relationships."""
        if not is_dataclass(node):
            if isinstance(node, (tuple, frozenset)):
                # Get hashes of immediate children
                child_hashes = [subtree_hash(elem) for elem in node]
                # Compute hash of current node
                own_hash = subtree_hash(node)
                # Map this node's hash to its children's hashes
                hash_map[own_hash] = frozenset(child_hashes)
                # Recurse into children
                for elem in node:
                    traverse(elem)
            else:
                # Leaf node: no children
                own_hash = subtree_hash(node)
                hash_map[own_hash] = frozenset()
        else:
            # Dataclass: process fields as children
            child_hashes = [
                subtree_hash(getattr(node, field.name))
                for field in fields(node)
            ]
            own_hash = subtree_hash(node)
            hash_map[own_hash] = frozenset(child_hashes)
            # Recurse into field values
            for field in fields(node):
                traverse(getattr(node, field.name))

    traverse(obj)
    return hash_map


# Tests


def test_traversals():
    def after(node):
        return node.children

    # Test 1: None root
    assert list(breadth_first_preorder(after, None)) == [], (
        "BF Preorder failed with None root"
    )
    assert list(breadth_first_postorder(after, None)) == [], (
        "BF Postorder failed with None root"
    )
    assert list(depth_first_preorder(after, None)) == [], (
        "DF Preorder failed with None root"
    )
    assert list(depth_first_postorder(after, None)) == [], (
        "DF Postorder failed with None root"
    )

    # Test 2: Single node
    single_node = RoseNode("A")
    assert [
        node.value for node in breadth_first_preorder(after, single_node)
    ] == ["A"], "BF Preorder failed with single node"
    assert [
        node.value for node in breadth_first_postorder(after, single_node)
    ] == ["A"], "BF Postorder failed with single node"
    assert [
        node.value for node in depth_first_preorder(after, single_node)
    ] == ["A"], "DF Preorder failed with single node"
    assert [
        node.value for node in depth_first_postorder(after, single_node)
    ] == ["A"], "DF Postorder failed with single node"

    # Test 3: Multi-level tree
    # Construct the tree:
    #       A
    #      / \
    #     B   C
    #    / \   \
    #   D   E   F
    F = RoseNode("F")
    E = RoseNode("E")
    D = RoseNode("D")
    C = RoseNode("C", (F,))
    B = RoseNode("B", (D, E))
    A = RoseNode("A", (B, C))

    # Define expected orders based on traversal definitions
    expected_bf_pre = ["A", "B", "C", "D", "E", "F"]  # Level by level from root
    expected_bf_post = [
        "D",
        "E",
        "F",
        "B",
        "C",
        "A",
    ]  # Level by level from leaves to root
    expected_df_pre = [
        "A",
        "B",
        "D",
        "E",
        "C",
        "F",
    ]  # Node before children, depth-first
    expected_df_post = [
        "D",
        "E",
        "B",
        "F",
        "C",
        "A",
    ]  # Children before node, depth-first

    # Perform assertions
    assert [
        node.value for node in breadth_first_preorder(after, A)
    ] == expected_bf_pre, "BF Preorder failed with sample tree"
    assert [
        node.value for node in breadth_first_postorder(after, A)
    ] == expected_bf_post, "BF Postorder failed with sample tree"
    assert [
        node.value for node in depth_first_preorder(after, A)
    ] == expected_df_pre, "DF Preorder failed with sample tree"
    assert [
        node.value for node in depth_first_postorder(after, A)
    ] == expected_df_post, "DF Postorder failed with sample tree"


# Test function for preorder_map and postorder_map
def test_mappings():
    # Construct the sample tree:
    #       A
    #      / \
    #     B   C
    #    / \   \
    #   D   E   F
    F = RoseNode("F")
    E = RoseNode("E")
    D = RoseNode("D")
    C = RoseNode("C", (F,))
    B = RoseNode("B", (D, E))
    A = RoseNode("A", (B, C))

    # Define helper functions
    def after(node: RoseNode[T]) -> Iterator[RoseNode[T]]:
        """Returns an iterator over the node's children."""
        return iter(node.children)

    def reconstruct(
        node: "RoseNode[T]", new_children: Sequence["RoseNode[T]"]
    ) -> "RoseNode[T]":
        """
        Reconstructs a node with its value and new children.

        Args:
            node: Original node providing the value.
            new_children: New list of child nodes.

        Returns:
            RoseNode: New node with node's value and new_children.
        """
        new_node = RoseNode(node.value, tuple(new_children))
        return new_node

    # Test Case 1: Transformation appending "_transformed"
    def f_transform(node: RoseNode[str]) -> RoseNode[str]:
        """Transforms node by appending '_transformed' to its value."""
        new_node = RoseNode(node.value + "_transformed", node.children)
        return new_node

    # Expected tree after transformation
    F_trans = RoseNode("F_transformed")
    E_trans = RoseNode("E_transformed")
    D_trans = RoseNode("D_transformed")
    C_trans = RoseNode("C_transformed", (F_trans,))
    B_trans = RoseNode("B_transformed", (D_trans, E_trans))
    expected_transform = RoseNode("A_transformed", (B_trans, C_trans))

    # Apply preorder_map
    preorder_result_transform = preorder_map(A, f_transform, after, reconstruct)
    assert preorder_result_transform == expected_transform, (
        "preorder_map with transformation failed"
    )

    # Apply postorder_map
    postorder_result_transform = postorder_map(
        A, f_transform, after, reconstruct
    )
    assert postorder_result_transform == expected_transform, (
        "postorder_map with transformation failed"
    )

    # Test Case 2: Identity transformation
    def f_identity(node: RoseNode[T]) -> RoseNode[T]:
        """Returns the node unchanged."""
        return node

    # Apply preorder_map with identity
    preorder_result_identity = preorder_map(A, f_identity, after, reconstruct)
    assert preorder_result_identity == A, "preorder_map with identity failed"

    # Apply postorder_map with identity
    postorder_result_identity = postorder_map(A, f_identity, after, reconstruct)
    assert postorder_result_identity == A, "postorder_map with identity failed"


def test_hash_hierarchy():
    # Original tree test (without shared subtrees)
    F = RoseNode("F")
    E = RoseNode("E")
    D = RoseNode("D")
    C = RoseNode("C", (F,))
    B = RoseNode("B", (D, E))
    A = RoseNode("A", (B, C))

    # Compute hashes for the original tree
    hash_A = subtree_hash(A)
    hash_B = subtree_hash(B)
    hash_C = subtree_hash(C)
    hash_D = subtree_hash(D)
    hash_E = subtree_hash(E)
    hash_F = subtree_hash(F)

    hash_str_A = subtree_hash("A")
    hash_str_B = subtree_hash("B")
    hash_str_C = subtree_hash("C")
    hash_str_D = subtree_hash("D")
    hash_str_E = subtree_hash("E")
    hash_str_F = subtree_hash("F")

    hash_tuple_BC = subtree_hash(A.children)  # (B, C)
    hash_tuple_DE = subtree_hash(B.children)  # (D, E)
    hash_tuple_F = subtree_hash(C.children)  # (F,)
    hash_empty = subtree_hash(())  # ()

    # Build the hash hierarchy for the original tree
    hash_map = build_hash_hierarchy(A)

    # Verify mappings for the original tree
    assert hash_map[hash_A] == frozenset([hash_str_A, hash_tuple_BC]), (
        "A mapping incorrect"
    )
    assert hash_map[hash_B] == frozenset([hash_str_B, hash_tuple_DE]), (
        "B mapping incorrect"
    )
    assert hash_map[hash_C] == frozenset([hash_str_C, hash_tuple_F]), (
        "C mapping incorrect"
    )
    assert hash_map[hash_D] == frozenset([hash_str_D, hash_empty]), (
        "D mapping incorrect"
    )
    assert hash_map[hash_E] == frozenset([hash_str_E, hash_empty]), (
        "E mapping incorrect"
    )
    assert hash_map[hash_F] == frozenset([hash_str_F, hash_empty]), (
        "F mapping incorrect"
    )

    assert hash_map[hash_tuple_BC] == frozenset([hash_B, hash_C]), (
        "Tuple (B,C) mapping incorrect"
    )
    assert hash_map[hash_tuple_DE] == frozenset([hash_D, hash_E]), (
        "Tuple (D,E) mapping incorrect"
    )
    assert hash_map[hash_tuple_F] == frozenset([hash_F]), (
        "Tuple (F,) mapping incorrect"
    )
    assert hash_map[hash_empty] == frozenset(), "Empty tuple mapping incorrect"

    assert hash_map[hash_str_A] == frozenset(), "String 'A' mapping incorrect"
    assert hash_map[hash_str_B] == frozenset(), "String 'B' mapping incorrect"
    assert hash_map[hash_str_C] == frozenset(), "String 'C' mapping incorrect"
    assert hash_map[hash_str_D] == frozenset(), "String 'D' mapping incorrect"
    assert hash_map[hash_str_E] == frozenset(), "String 'E' mapping incorrect"
    assert hash_map[hash_str_F] == frozenset(), "String 'F' mapping incorrect"

    # Verify no extra or missing keys in the original tree
    expected_keys = {
        hash_A,
        hash_B,
        hash_C,
        hash_D,
        hash_E,
        hash_F,
        hash_tuple_BC,
        hash_tuple_DE,
        hash_tuple_F,
        hash_empty,
        hash_str_A,
        hash_str_B,
        hash_str_C,
        hash_str_D,
        hash_str_E,
        hash_str_F,
    }
    assert set(hash_map.keys()) == expected_keys, (
        "Extra or missing keys in hash_map"
    )

    # New test case with shared subtrees
    G = RoseNode("G")
    # Reuse existing D, E, F, and create new B and C with G as a shared child
    B_shared = RoseNode("B", (D, E, G))
    C_shared = RoseNode("C", (F, G))
    A_shared = RoseNode("A", (B_shared, C_shared))

    # Compute hashes for the tree with shared subtrees
    hash_A_shared = subtree_hash(A_shared)
    hash_B_shared = subtree_hash(B_shared)
    hash_C_shared = subtree_hash(C_shared)
    hash_G = subtree_hash(G)

    hash_str_G = subtree_hash("G")

    hash_tuple_BC_shared = subtree_hash(
        A_shared.children
    )  # (B_shared, C_shared)
    hash_tuple_DEG = subtree_hash(B_shared.children)  # (D, E, G)
    hash_tuple_FG = subtree_hash(C_shared.children)  # (F, G)

    # Build the hash hierarchy for the tree with shared subtrees
    hash_map_shared = build_hash_hierarchy(A_shared)

    # Verify mappings for the tree with shared subtrees
    assert hash_map_shared[hash_A_shared] == frozenset(
        [hash_str_A, hash_tuple_BC_shared]
    ), "A_shared mapping incorrect"
    assert hash_map_shared[hash_B_shared] == frozenset(
        [hash_str_B, hash_tuple_DEG]
    ), "B_shared mapping incorrect"
    assert hash_map_shared[hash_C_shared] == frozenset(
        [hash_str_C, hash_tuple_FG]
    ), "C_shared mapping incorrect"
    assert hash_map_shared[hash_D] == frozenset([hash_str_D, hash_empty]), (
        "D mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_E] == frozenset([hash_str_E, hash_empty]), (
        "E mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_F] == frozenset([hash_str_F, hash_empty]), (
        "F mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_G] == frozenset([hash_str_G, hash_empty]), (
        "G mapping incorrect"
    )

    assert hash_map_shared[hash_tuple_BC_shared] == frozenset(
        [hash_B_shared, hash_C_shared]
    ), "Tuple (B_shared, C_shared) mapping incorrect"
    assert hash_map_shared[hash_tuple_DEG] == frozenset(
        [hash_D, hash_E, hash_G]
    ), "Tuple (D, E, G) mapping incorrect"
    assert hash_map_shared[hash_tuple_FG] == frozenset([hash_F, hash_G]), (
        "Tuple (F, G) mapping incorrect"
    )
    assert hash_map_shared[hash_empty] == frozenset(), (
        "Empty tuple mapping incorrect in shared tree"
    )

    assert hash_map_shared[hash_str_A] == frozenset(), (
        "String 'A' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_B] == frozenset(), (
        "String 'B' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_C] == frozenset(), (
        "String 'C' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_D] == frozenset(), (
        "String 'D' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_E] == frozenset(), (
        "String 'E' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_F] == frozenset(), (
        "String 'F' mapping incorrect in shared tree"
    )
    assert hash_map_shared[hash_str_G] == frozenset(), (
        "String 'G' mapping incorrect"
    )

    # Verify that hash_G is correctly shared
    assert hash_G in hash_map_shared[hash_tuple_DEG], (
        "hash_G missing from children of hash_tuple_DEG"
    )
    assert hash_G in hash_map_shared[hash_tuple_FG], (
        "hash_G missing from children of hash_tuple_FG"
    )

    # Verify no extra or missing keys in the shared tree
    expected_keys_shared = {
        hash_A_shared,
        hash_B_shared,
        hash_C_shared,
        hash_D,
        hash_E,
        hash_F,
        hash_G,
        hash_tuple_BC_shared,
        hash_tuple_DEG,
        hash_tuple_FG,
        hash_empty,
        hash_str_A,
        hash_str_B,
        hash_str_C,
        hash_str_D,
        hash_str_E,
        hash_str_F,
        hash_str_G,
    }
    assert set(hash_map_shared.keys()) == expected_keys_shared, (
        "Extra or missing keys in hash_map_shared"
    )

    print("All tests for hash hierarchy passed successfully!")


if __name__ == "__main__":
    test_traversals()
    test_mappings()
    test_hash_hierarchy()
    print("All tests passed successfully!")
