"""Tests for utils/tree_functionals.py"""

import pytest
from typing import Iterator, Sequence

from utils.tree_functionals import (
    RoseNode,
    breadth_first_preorder,
    breadth_first_postorder,
    depth_first_preorder,
    depth_first_postorder,
    postorder_map,
    preorder_map,
    cached_hash,
    build_hash_hierarchy,
)


def after(node: RoseNode) -> Iterator[RoseNode]:
    """Returns iterator over node's children."""
    return iter(node.children)


def reconstruct(node: RoseNode, new_children: Sequence[RoseNode]) -> RoseNode:
    """Reconstructs node with new children."""
    return RoseNode(node.value, tuple(new_children))


class TestTraversals:
    def test_none_root(self):
        assert list(breadth_first_preorder(after, None)) == []
        assert list(breadth_first_postorder(after, None)) == []
        assert list(depth_first_preorder(after, None)) == []
        assert list(depth_first_postorder(after, None)) == []

    def test_single_node(self):
        single = RoseNode("A")
        assert [n.value for n in breadth_first_preorder(after, single)] == ["A"]
        assert [n.value for n in breadth_first_postorder(after, single)] == ["A"]
        assert [n.value for n in depth_first_preorder(after, single)] == ["A"]
        assert [n.value for n in depth_first_postorder(after, single)] == ["A"]

    def test_multi_level_tree(self):
        """
        Tree structure:
              A
             / \\
            B   C
           / \\   \\
          D   E   F
        """
        F = RoseNode("F")
        E = RoseNode("E")
        D = RoseNode("D")
        C = RoseNode("C", (F,))
        B = RoseNode("B", (D, E))
        A = RoseNode("A", (B, C))

        # BFS preorder: level by level from root
        assert [n.value for n in breadth_first_preorder(after, A)] == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]

        # BFS postorder: level by level from leaves
        assert [n.value for n in breadth_first_postorder(after, A)] == [
            "D",
            "E",
            "F",
            "B",
            "C",
            "A",
        ]

        # DFS preorder: parent before children
        assert [n.value for n in depth_first_preorder(after, A)] == [
            "A",
            "B",
            "D",
            "E",
            "C",
            "F",
        ]

        # DFS postorder: children before parent
        assert [n.value for n in depth_first_postorder(after, A)] == [
            "D",
            "E",
            "B",
            "F",
            "C",
            "A",
        ]


class TestMappings:
    def test_transform(self):
        """Test transformation that appends '_t' to each value."""
        F = RoseNode("F")
        E = RoseNode("E")
        D = RoseNode("D")
        C = RoseNode("C", (F,))
        B = RoseNode("B", (D, E))
        A = RoseNode("A", (B, C))

        def transform(node: RoseNode) -> RoseNode:
            return RoseNode(node.value + "_t", node.children)

        # Expected result
        F_t = RoseNode("F_t")
        E_t = RoseNode("E_t")
        D_t = RoseNode("D_t")
        C_t = RoseNode("C_t", (F_t,))
        B_t = RoseNode("B_t", (D_t, E_t))
        expected = RoseNode("A_t", (B_t, C_t))

        assert preorder_map(A, transform, after, reconstruct) == expected
        assert postorder_map(A, transform, after, reconstruct) == expected

    def test_identity(self):
        """Identity transformation should preserve structure."""
        F = RoseNode("F")
        E = RoseNode("E")
        D = RoseNode("D")
        C = RoseNode("C", (F,))
        B = RoseNode("B", (D, E))
        A = RoseNode("A", (B, C))

        identity = lambda n: n

        assert preorder_map(A, identity, after, reconstruct) == A
        assert postorder_map(A, identity, after, reconstruct) == A


class TestHashHierarchy:
    def test_simple_tree(self):
        F = RoseNode("F")
        E = RoseNode("E")
        D = RoseNode("D")
        C = RoseNode("C", (F,))
        B = RoseNode("B", (D, E))
        A = RoseNode("A", (B, C))

        hash_map = build_hash_hierarchy(A)

        # Verify all nodes are in the map
        assert cached_hash(A) in hash_map
        assert cached_hash(B) in hash_map
        assert cached_hash(C) in hash_map
        assert cached_hash(D) in hash_map
        assert cached_hash(E) in hash_map
        assert cached_hash(F) in hash_map

        # Verify leaf strings have no children
        assert hash_map[cached_hash("A")] == frozenset()
        assert hash_map[cached_hash("F")] == frozenset()

    def test_shared_subtrees(self):
        """Test tree where same node appears in multiple places."""
        G = RoseNode("G")
        F = RoseNode("F")
        E = RoseNode("E")
        D = RoseNode("D")
        # G is shared between B and C
        B = RoseNode("B", (D, E, G))
        C = RoseNode("C", (F, G))
        A = RoseNode("A", (B, C))

        hash_map = build_hash_hierarchy(A)

        # G should appear once in the map
        assert cached_hash(G) in hash_map
        assert cached_hash("G") in hash_map
