"""
Tests for the hierarchy package.

Tests component extraction, DAG building, and syntax tree construction.
"""

from collections import defaultdict

import pytest

from hierarchy import (
    components_by_colors_to_grid_object_dag,
    condition_by_color_couples,
    condition_by_colors,
    coords_to_connected_components,
    grid_to_components_by_colors,
)
from localtypes import Coord, GridObject


class TestCoordsToConnectedComponents:
    """Tests for coords_to_connected_components (8-connectivity)."""

    def test_empty_set(self):
        result = coords_to_connected_components(set())
        assert result == frozenset()

    def test_single_coord(self):
        result = coords_to_connected_components({(0, 0)})
        assert result == frozenset([frozenset({(0, 0)})])

    def test_connected_l_shape(self):
        coords = {(0, 0), (1, 0), (0, 1)}
        result = coords_to_connected_components(coords)
        expected = frozenset([frozenset({(0, 0), (1, 0), (0, 1)})])
        assert result == expected

    def test_two_disconnected(self):
        coords = {(0, 0), (2, 2)}
        result = coords_to_connected_components(coords)
        expected = frozenset([frozenset({(0, 0)}), frozenset({(2, 2)})])
        assert result == expected

    def test_two_separate_lines(self):
        coords = {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)}
        result = coords_to_connected_components(coords)
        expected = frozenset([
            frozenset({(0, 0), (0, 1), (0, 2)}),
            frozenset({(2, 0), (2, 1), (2, 2)}),
        ])
        assert result == expected

    def test_diagonal_connectivity(self):
        """8-connectivity should connect diagonal neighbors."""
        coords = {(0, 0), (1, 1)}
        result = coords_to_connected_components(coords)
        expected = frozenset([frozenset({(0, 0), (1, 1)})])
        assert result == expected

    def test_all_corners_connected(self):
        """All 5 coords should be connected via diagonals."""
        coords = {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}
        result = coords_to_connected_components(coords)
        expected = frozenset([frozenset({(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)})])
        assert result == expected


class TestConditionByColorCouples:
    """Tests for condition_by_color_couples."""

    def test_2x2_three_colors(self):
        grid = [[0, 1], [2, 0]]
        expected = {
            (0, 0): {(0, 0), (1, 1)},
            (1, 0): {(0, 0), (1, 0), (1, 1)},
            (1, 1): {(1, 0)},
            (2, 0): {(0, 0), (0, 1), (1, 1)},
            (2, 1): {(0, 1), (1, 0)},
            (2, 2): {(0, 1)},
        }
        result = condition_by_color_couples(grid)
        assert result == expected

    def test_2x2_single_color(self):
        grid = [[0, 0], [0, 0]]
        expected = {(0, 0): {(0, 0), (1, 0), (0, 1), (1, 1)}}
        result = condition_by_color_couples(grid)
        assert result == expected

    def test_2x3_three_colors(self):
        grid = [[0, 1, 2], [2, 1, 0]]
        expected = {
            (0, 0): {(0, 0), (2, 1)},
            (1, 0): {(0, 0), (1, 0), (1, 1), (2, 1)},
            (1, 1): {(1, 0), (1, 1)},
            (2, 0): {(0, 0), (0, 1), (2, 0), (2, 1)},
            (2, 1): {(0, 1), (1, 0), (1, 1), (2, 0)},
            (2, 2): {(2, 0), (0, 1)},
        }
        result = condition_by_color_couples(grid)
        assert result == expected


class TestConditionByColors:
    """Tests for condition_by_colors."""

    def test_2x2_three_colors(self):
        grid = [[0, 1], [2, 0]]
        result = condition_by_colors(grid)

        assert result[frozenset({0})] == {(0, 0), (1, 1)}
        assert result[frozenset({1})] == {(1, 0)}
        assert result[frozenset({2})] == {(0, 1)}
        assert result[frozenset({0, 1})] == {(0, 0), (1, 0), (1, 1)}
        assert result[frozenset({0, 2})] == {(0, 0), (0, 1), (1, 1)}
        assert result[frozenset({1, 2})] == {(1, 0), (0, 1)}
        assert result[frozenset({0, 1, 2})] == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert len(result) == 7  # 2^3 - 1


class TestGridToComponentsByColors:
    """Tests for grid_to_components_by_colors with 8-connectivity."""

    def test_2x2_three_colors(self):
        grid = [[0, 1], [2, 0]]
        result = grid_to_components_by_colors(grid)
        expected = defaultdict(set, {
            frozenset({0}): {frozenset({Coord(0, 0), Coord(1, 1)})},
            frozenset({1}): {frozenset({Coord(1, 0)})},
            frozenset({2}): {frozenset({Coord(0, 1)})},
            frozenset({0, 1}): {frozenset({Coord(0, 0), Coord(1, 0), Coord(1, 1)})},
            frozenset({0, 2}): {frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 1)})},
            frozenset({1, 2}): {frozenset({Coord(1, 0), Coord(0, 1)})},
            frozenset({0, 1, 2}): {frozenset({Coord(0, 0), Coord(1, 0), Coord(0, 1), Coord(1, 1)})},
        })

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            assert result[key] == expected[key], f"Mismatch for key {key}"

    def test_1x1_single_color(self):
        grid = [[0]]
        result = grid_to_components_by_colors(grid)
        expected = defaultdict(set, {frozenset({0}): {frozenset({Coord(0, 0)})}})

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            assert result[key] == expected[key]

    def test_2x2_diagonal_pattern(self):
        grid = [[0, 1], [1, 0]]
        result = grid_to_components_by_colors(grid)
        expected = defaultdict(set, {
            frozenset({0}): {frozenset({Coord(0, 0), Coord(1, 1)})},
            frozenset({1}): {frozenset({Coord(0, 1), Coord(1, 0)})},
            frozenset({0, 1}): {frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)})},
        })

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            assert result[key] == expected[key]

    def test_2x2_single_color(self):
        grid = [[0, 0], [0, 0]]
        result = grid_to_components_by_colors(grid)
        expected = defaultdict(set, {
            frozenset({0}): {frozenset({Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)})}
        })

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            assert result[key] == expected[key]


class TestComponentsByColorsToGridObjectDag:
    """Tests for components_by_colors_to_grid_object_dag."""

    def test_simple_lattice(self):
        """Test a simple 3-element lattice."""
        components_by_colors: dict = {
            frozenset({0}): frozenset({frozenset({Coord(0, 0)})}),
            frozenset({1}): frozenset({frozenset({Coord(1, 1)})}),
            frozenset({2}): frozenset({frozenset({Coord(2, 2)})}),
            frozenset({0, 1}): frozenset({frozenset({Coord(0, 0), Coord(1, 1)})}),
            frozenset({0, 2}): frozenset({frozenset({Coord(0, 0), Coord(2, 2)})}),
            frozenset({1, 2}): frozenset({frozenset({Coord(1, 1), Coord(2, 2)})}),
            frozenset({0, 1, 2}): frozenset({frozenset({Coord(0, 0), Coord(1, 1), Coord(2, 2)})}),
        }

        result = components_by_colors_to_grid_object_dag(components_by_colors)

        # Define expected GridObjects
        GO0 = GridObject(frozenset({0}), frozenset({Coord(0, 0)}))
        GO1 = GridObject(frozenset({1}), frozenset({Coord(1, 1)}))
        GO2 = GridObject(frozenset({2}), frozenset({Coord(2, 2)}))
        GO01 = GridObject(frozenset({0, 1}), frozenset({Coord(0, 0), Coord(1, 1)}))
        GO02 = GridObject(frozenset({0, 2}), frozenset({Coord(0, 0), Coord(2, 2)}))
        GO12 = GridObject(frozenset({1, 2}), frozenset({Coord(1, 1), Coord(2, 2)}))
        GO012 = GridObject(frozenset({0, 1, 2}), frozenset({Coord(0, 0), Coord(1, 1), Coord(2, 2)}))

        expected = {
            GO012: {GO01, GO02, GO12},
            GO01: {GO0, GO1},
            GO02: {GO0, GO2},
            GO12: {GO1, GO2},
            GO0: set(),
            GO1: set(),
            GO2: set(),
        }

        assert set(result.keys()) == set(expected.keys()), "DAG keys mismatch"
        for key in expected:
            assert result[key] == expected[key], f"DAG mismatch for {key}"
