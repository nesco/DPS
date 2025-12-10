"""Tests for utils/algorithms/dag.py"""

import pytest
from utils.algorithms.dag import topological_sort


class TestTopologicalSort:
    def test_linear_chain(self):
        """A -> B -> C should give [A, B, C]"""
        graph = {"A": {"B"}, "B": {"C"}, "C": set()}
        result = topological_sort(graph)
        assert list(result) == ["A", "B", "C"]

    def test_diamond(self):
        """
        Diamond: A -> B, A -> C, B -> D, C -> D
        Valid orders: [A, B, C, D] or [A, C, B, D]
        """
        graph = {"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": set()}
        result = topological_sort(graph)

        # Verify A comes before B, C; B and C come before D
        result_list = list(result)
        assert result_list.index("A") < result_list.index("B")
        assert result_list.index("A") < result_list.index("C")
        assert result_list.index("B") < result_list.index("D")
        assert result_list.index("C") < result_list.index("D")

    def test_disconnected_nodes(self):
        """Nodes with no edges should still be included."""
        graph = {"A": set(), "B": set(), "C": set()}
        result = topological_sort(graph)
        assert set(result) == {"A", "B", "C"}

    def test_child_only_nodes(self):
        """Nodes that only appear as children (not keys) should be included."""
        graph = {"A": {"B", "C"}}  # B and C not in keys
        result = topological_sort(graph)
        assert set(result) == {"A", "B", "C"}
        assert list(result).index("A") < list(result).index("B")
        assert list(result).index("A") < list(result).index("C")

    def test_empty_graph(self):
        """Empty graph should return empty tuple."""
        graph: dict[str, set[str]] = {}
        result = topological_sort(graph)
        assert result == ()

    def test_single_node(self):
        """Single node with no edges."""
        graph = {"A": set()}
        result = topological_sort(graph)
        assert result == ("A",)

    def test_cycle_detection(self):
        """Graph with cycle should raise ValueError."""
        graph = {"A": {"B"}, "B": {"C"}, "C": {"A"}}
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(graph)

    def test_self_loop_detection(self):
        """Node with self-loop should raise ValueError."""
        graph = {"A": {"A"}}
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(graph)

    def test_complex_dag(self):
        """
        More complex DAG:
        1 -> 2, 3
        2 -> 4
        3 -> 4, 5
        4 -> 6
        5 -> 6
        """
        graph = {
            1: {2, 3},
            2: {4},
            3: {4, 5},
            4: {6},
            5: {6},
            6: set(),
        }
        result = topological_sort(graph)
        result_list = list(result)

        # Verify all ordering constraints
        assert result_list.index(1) < result_list.index(2)
        assert result_list.index(1) < result_list.index(3)
        assert result_list.index(2) < result_list.index(4)
        assert result_list.index(3) < result_list.index(4)
        assert result_list.index(3) < result_list.index(5)
        assert result_list.index(4) < result_list.index(6)
        assert result_list.index(5) < result_list.index(6)
