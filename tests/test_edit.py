"""
Tests for the edit distance module.
"""

from dataclasses import dataclass, field

from edit import (
    Add,
    Delete,
    Identity,
    Substitute,
    Inner,
    Prune,
    Graft,
    sequence_edit_distance,
    set_edit_distance,
    recursive_edit_distance,
    extended_edit_distance,
    apply_transformation,
)
from kolmogorov_tree.types import BitLengthAware, KeyValue, Primitive
from utils.algorithms.tree import (
    RoseNode,
    breadth_first_preorder_bitlengthaware,
)


@dataclass(frozen=True)
class MockValue(Primitive):
    value: int

    def bit_length(self) -> int:
        return 1


@dataclass(frozen=True)
class CharValue(Primitive):
    value: str

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


def test_edit_distance():
    """Tests the edit distance computation for strings with MDL cost model.

    MDL costs:
    - Delete: log2(source_length) to specify position
    - Add: element.bit_length() to describe content
    - Substitute: target.bit_length() (receiver knows source)
    """
    a = "kitten"
    b = "sitting"

    a_tuple = tuple(CharValue(c) for c in a)
    b_tuple = tuple(CharValue(c) for c in b)

    distance, transformation = sequence_edit_distance(
        a_tuple,
        b_tuple,
        distance_func=lambda x, y: (0, Identity(KeyValue(None)))
        if x == y
        else (1, Substitute(KeyValue(None), y)),  # substitute costs 1 (target only)
        len_func=lambda x: 1,
    )
    # With MDL: delete costs log2(6)=3 bits, add costs 1, substitute costs 1
    # Best strategy: substitute k→s, i=i, t=t, t=t, substitute e→i, n=n, add g
    # Cost: 1 + 0 + 0 + 0 + 1 + 0 + 1 = 3
    assert distance == 3, f"Expected distance 3, got {distance}"


def test_set_edit_distance():
    """Tests the edit distance computation for sets."""
    source_set = frozenset({CharValue("a"), CharValue("b")})
    target_set = frozenset({CharValue("a"), CharValue("c")})
    distance, transformation = set_edit_distance(
        source_set,
        target_set,
        distance_func=lambda x, y: (0, Identity(KeyValue(None)))
        if x == y
        else (2, Substitute(KeyValue(None), y)),
        len_func=lambda x: 1,
    )
    assert distance == 2, f"Expected distance 2, got {distance}"
    # Should have: Identity for 'a', Delete for 'b', Add for 'c'
    assert len(transformation) == 3, f"Expected 3 operations, got {len(transformation)}"


def test_recursive_edit_distance():
    """Tests recursive edit distance for tree structures."""
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
    root2 = TreeNode(prim1, (leaf1, leaf2))  # Tree: 1 -> (1, 2), identical to root1
    root3 = TreeNode(prim1, (leaf1, leaf3))  # Tree: 1 -> (1, 3)
    root4 = TreeNode(prim1, (leaf1, leaf2, leaf4))  # Tree: 1 -> (1, 2, 4)
    root5 = TreeNode(prim5)  # Tree: 5

    # Test Case 1: Identical Trees
    distance, transformation = recursive_edit_distance(root1, root2)
    assert distance == 0, "Distance should be 0 for identical trees"
    assert isinstance(transformation, Identity), "Transformation should be Identity"

    # Test Case 2: One Leaf Value Changed
    distance, transformation = recursive_edit_distance(root1, root3)
    # MDL: substitute costs target.bit_length() = 1
    assert distance == 1, (
        "Distance should be 1 (substitute leaf2.value with leaf3.value)"
    )
    assert any(
        isinstance(operation, Substitute)
        for operation in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Substitute"

    # Test Case 3: Added Node
    distance, transformation = recursive_edit_distance(root1, root4)
    assert distance == 1, "Distance should be 1 (add leaf4)"
    assert any(
        isinstance(op, Add)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Add"

    # Test Case 4: Completely Different Trees
    distance, transformation = recursive_edit_distance(root1, root5)
    # MDL: substitute value costs 1, delete 2 children costs log2(2)=1 each = 2
    # Total: 1 + 2 = 3
    assert distance == 3, (
        "Distance should be 3 (substitute value + delete two children)"
    )
    assert any(
        isinstance(op, Substitute)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Substitute"
    assert any(
        isinstance(op, Delete)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Should include Delete"


def test_extended_edit_distance():
    """Tests extended edit distance with Prune and Graft operations."""
    # Create leaf nodes
    leaf_E = TreeNode(MockValue(3))
    leaf_D = TreeNode(MockValue(3))
    leaf_C = TreeNode(MockValue(2))

    node_B = TreeNode(MockValue(0), (leaf_C, leaf_D))
    node_A = TreeNode(MockValue(1), (node_B, leaf_E))

    # Test 1: Prune operation
    distance, transformation = extended_edit_distance(node_A, node_B, tuple())
    assert transformation is not None, "The transformation should not be None"
    assert distance == 2, f"Expected distance 2 for Prune, got {distance}"
    assert any(
        isinstance(op, Prune)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Transformations should include Prune"

    # Test 2: Graft operation
    distance, transformation = extended_edit_distance(node_B, node_A, tuple())
    assert transformation is not None, "The transformation should not be None"
    assert distance == 2, f"Expected distance 2 for Graft, got {distance}"
    assert any(
        isinstance(op, Graft)
        for op in breadth_first_preorder_bitlengthaware(transformation)
    ), "Transformations should include Graft"


def test_apply_transformations():
    """Tests the application of transformations."""

    @dataclass(frozen=True)
    class TupleWrapper(BitLengthAware):
        items: tuple[MockValue, ...]

        def bit_length(self) -> int:
            return sum(item.bit_length() for item in self.items)

        def __str__(self) -> str:
            return f"TupleWrapper({', '.join(str(item) for item in self.items)})"

    # Test Case 1: Simple Primitive Substitution
    source_1 = MockValue(1)
    target_1 = MockValue(2)
    _, transform_1 = recursive_edit_distance(source_1, target_1)
    result_1 = apply_transformation(source_1, transform_1)
    assert result_1 == target_1, f"Test 1 failed: {result_1} != {target_1}"

    # Test Case 2: Tuple
    source_2 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3)))
    target_2 = TupleWrapper((MockValue(1), MockValue(4), MockValue(3)))
    _, transform_2 = recursive_edit_distance(source_2, target_2)
    result_2 = apply_transformation(source_2, transform_2)
    assert result_2 == target_2, f"Test 2 failed: {result_2} != {target_2}"

    # Test Case 3: Dataclass with Tuple Field
    source_3 = TreeNode(MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(3))))
    target_3 = TreeNode(MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(4))))
    _, transform_3 = recursive_edit_distance(source_3, target_3)
    result_3 = apply_transformation(source_3, transform_3)
    assert result_3 == target_3, f"Test 3 failed: {result_3} != {target_3}"

    # Test Case 4: Prune Operation
    leaf_E = TreeNode(MockValue(3))
    leaf_D = TreeNode(MockValue(3))
    leaf_C = TreeNode(MockValue(2))
    node_B = TreeNode(MockValue(0), (leaf_C, leaf_D))
    node_A = TreeNode(MockValue(1), (node_B, leaf_E))
    source_4 = node_A
    target_4 = node_B
    _, transform_4 = extended_edit_distance(source_4, target_4, tuple())
    assert transform_4 is not None, "The transformation should not be None"
    result_4 = apply_transformation(source_4, transform_4)
    assert result_4 == target_4, f"Test 4 failed: {result_4} != {target_4}"

    # Test Case 5: Graft Operation
    source_5 = node_B
    target_5 = node_A
    _, transform_5 = extended_edit_distance(source_5, target_5, tuple())
    assert transform_5 is not None, "The transformation should not be None"
    result_5 = apply_transformation(source_5, transform_5)
    assert result_5 == target_5, f"Test 5 failed: {result_5} != {target_5}"


def test_tuple_deletion_roundtrip():
    """Tests that deletions from tuples can be computed and applied correctly.
    
    This is a regression test for the issue where sequence_edit_distance()
    produces Delete operations at arbitrary indices, but apply_transformation()
    previously only allowed deletion at the last index.
    """

    @dataclass(frozen=True)
    class TupleWrapper(BitLengthAware):
        items: tuple[MockValue, ...]

        def bit_length(self) -> int:
            return sum(item.bit_length() for item in self.items)

        def __str__(self) -> str:
            return f"TupleWrapper({', '.join(str(item) for item in self.items)})"

    # Test Case 1: Delete middle element
    # Source: (1, 2, 3) -> Target: (1, 3)
    source_1 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3)))
    target_1 = TupleWrapper((MockValue(1), MockValue(3)))
    _, transform_1 = recursive_edit_distance(source_1, target_1)
    result_1 = apply_transformation(source_1, transform_1)
    assert result_1 == target_1, f"Middle deletion failed: {result_1} != {target_1}"

    # Test Case 2: Delete first element
    # Source: (1, 2, 3) -> Target: (2, 3)
    source_2 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3)))
    target_2 = TupleWrapper((MockValue(2), MockValue(3)))
    _, transform_2 = recursive_edit_distance(source_2, target_2)
    result_2 = apply_transformation(source_2, transform_2)
    assert result_2 == target_2, f"First element deletion failed: {result_2} != {target_2}"

    # Test Case 3: Delete multiple elements (keep only first and last)
    # Source: (1, 2, 3, 4) -> Target: (1, 4)
    source_3 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3), MockValue(4)))
    target_3 = TupleWrapper((MockValue(1), MockValue(4)))
    _, transform_3 = recursive_edit_distance(source_3, target_3)
    result_3 = apply_transformation(source_3, transform_3)
    assert result_3 == target_3, f"Multiple deletion failed: {result_3} != {target_3}"

    # Test Case 4: Delete all but one (keep middle)
    # Source: (1, 2, 3) -> Target: (2,)
    source_4 = TupleWrapper((MockValue(1), MockValue(2), MockValue(3)))
    target_4 = TupleWrapper((MockValue(2),))
    _, transform_4 = recursive_edit_distance(source_4, target_4)
    result_4 = apply_transformation(source_4, transform_4)
    assert result_4 == target_4, f"Keep middle only failed: {result_4} != {target_4}"

    # Test Case 5: TreeNode children deletion (nested structure)
    source_5 = TreeNode(
        MockValue(1),
        (TreeNode(MockValue(2)), TreeNode(MockValue(3)), TreeNode(MockValue(4))),
    )
    target_5 = TreeNode(MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(4))))
    _, transform_5 = recursive_edit_distance(source_5, target_5)
    result_5 = apply_transformation(source_5, transform_5)
    assert result_5 == target_5, f"Nested deletion failed: {result_5} != {target_5}"


def test_duplicate_subtrees():
    """Tests that trees with structurally equal subtrees are handled correctly.

    This is a regression test for the issue where collect_links() used hash()
    for node identification, causing distinct but equal subtrees to be merged.
    """
    # Create a tree with two identical subtrees at different positions:
    #        root(1)
    #       /       \
    #    leaf(2)   leaf(2)   <- These are structurally equal but distinct nodes
    #
    leaf1 = TreeNode(MockValue(2))
    leaf2 = TreeNode(MockValue(2))  # Same value, different object

    # Verify they are equal but not identical
    assert leaf1 == leaf2, "Leaves should be structurally equal"
    assert leaf1 is not leaf2, "Leaves should be distinct objects"
    assert hash(leaf1) == hash(leaf2), "Equal objects should have equal hashes"

    source = TreeNode(MockValue(1), (leaf1, leaf2))

    # Target: change only the second child
    target = TreeNode(MockValue(1), (TreeNode(MockValue(2)), TreeNode(MockValue(3))))

    # This should produce a transformation that only affects the second child
    _, transform = recursive_edit_distance(source, target)
    result = apply_transformation(source, transform)
    assert result == target, f"Duplicate subtrees failed: {result} != {target}"

    # Also test with extended_edit_distance (which uses collect_links)
    _, ext_transform = extended_edit_distance(source, target, tuple())
    assert ext_transform is not None
    ext_result = apply_transformation(source, ext_transform)
    assert ext_result == target, f"Extended with duplicates failed: {ext_result} != {target}"


def test_prune_graft_with_duplicate_subtrees():
    """Tests prune/graft operations on trees with equal subtrees.

    Ensures the topology is correctly preserved when using id() for identification.
    """
    # Source tree:
    #        A(1)
    #       /    \
    #    B(2)    C(2)    <- B and C are equal
    #    |
    #  D(3)
    #
    leaf_D = TreeNode(MockValue(3))
    node_B = TreeNode(MockValue(2), (leaf_D,))
    node_C = TreeNode(MockValue(2))  # Equal to node_B's value but different structure
    source = TreeNode(MockValue(1), (node_B, node_C))

    # Target: just the leaf D (should use Prune to get to D)
    target = leaf_D

    dist, transform = extended_edit_distance(source, target, tuple())
    assert transform is not None, "Should find a transformation"
    result = apply_transformation(source, transform)
    assert result == target, f"Prune with duplicates failed: {result} != {target}"
