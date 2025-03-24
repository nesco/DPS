from utils.dag_functionals import topological_sort
from utils.graph import nodes_to_connected_components
from utils.grid import coords_to_proportions, grid_to_coords_by_color, grid_to_points, grid_to_proportions, DIRECTIONS
from localtypes import Color, Colors, Coord, Coords, ColorGrid, GridObject
from itertools import chain, combinations

from collections.abc import Mapping, Set

def coords_to_connected_components(coords: Coords) -> frozenset[Coords]:

    def node_to_neighbours(node: Coord) -> Coords:
        row, col = node
        possibilities = set(Coord(row+drow, col+dcol) for drow, dcol in DIRECTIONS.values())
        neighbours = possibilities.intersection(coords)
        return neighbours

    return nodes_to_connected_components(coords, node_to_neighbours)

def condition_by_color_couples(grid: ColorGrid) -> dict[tuple[Color, Color], Coords]:
    """
    Returns a dict of all masks comprised of union of two colours.
    TO-DO: N colors?
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = sorted(list(coords_by_color.keys()))

    coords_by_color_couple: dict[tuple[Color, Color], Coords] = {}

    for i in range(len(colors)):
        color_a = colors[i]
        for j in range(i):
            color_b = colors[j]
            coords_by_color_couple[(color_a, color_b)] = (
                coords_by_color[color_a] | coords_by_color[color_b]
            )

        coords_by_color_couple[(color_a, color_a)] = coords_by_color[color_a]
    return coords_by_color_couple

def condition_by_colors(grid: ColorGrid) -> dict[Colors, Coords]:
    """
    Returns a dictionary where each key is a frozenset of colors (representing a subset of colors)
    and each value is the set of coordinates where the color in the grid is in that subset.
    """
    coords_by_color = grid_to_coords_by_color(grid)
    colors = list(coords_by_color.keys())
    all_subsets = chain.from_iterable(combinations(colors, r) for r in range(1, len(colors) + 1))
    result = {}
    for subset in all_subsets:
        S = frozenset(subset)
        union_coords = set().union(*(coords_by_color[c] for c in S))
        result[S] = union_coords
    return result

def grid_to_components_by_colors(grid: ColorGrid) -> dict[Colors, frozenset[Coords]]:
    components_by_colors = {}
    coords_by_colors = condition_by_colors(grid)
    for colors, coords in coords_by_colors.items():
        components_by_colors[colors] = coords_to_connected_components(coords)

    return components_by_colors

def components_by_colors_to_grid_object_dag(components_by_colors: Mapping[Colors, Set[Coords]]) -> dict[GridObject, set[GridObject]]:
    """
    Transform the components into GridObjects, and build a DAG for the inclusion relation out of it, by building an Hass Diagram.
    In practice, the DAG acts a bit like a lattice, because selecting all the colors will end into a single connected component
    encompassing the entire grid.
    It will thus be the supremum of the inclusion relation, every other grid objects will be included into it.
    To form a full latice, the empty object with no colors nor coords should be included to, but will not for practicallity

    Args:
        components_by_colors: sets of components by the colors they are made of.

    Returns:
        dict[GridObject, set[GridObject]]: DAG of grid objects encoding the set inclusion relationship <=
    """

    # Step 1: Building Grid Objects
    grid_objects = tuple(GridObject(colors, component) for colors, components in components_by_colors.items() for component in components)

    # Step 2: Build the subsets and the supersets dict
    subsets = {a: frozenset(b for b in grid_objects if b.coords <= a.coords) for a in grid_objects}
    supersets = {b: frozenset(c for c in grid_objects if b.coords <= c.coords) for b in grid_objects}

    # Step 3: Initialize the DAG
    graph = {a: set() for a in grid_objects}

    # Step 4: For each set a, check all its proper subsets b
    for a in grid_objects:
        for b in subsets[a]:
            if b != a: # Ensure b is a proper subset of a
                intersection = subsets[a] & supersets[b]
                if len(intersection) == 2:  # Only a and b, no intermediate c
                    graph[a].add(b)

    return graph

def sort_by_inclusion(grid_object_dag: Mapping[GridObject, Set[GridObject]]) -> tuple[GridObject, ...]:
    return topological_sort(grid_object_dag)

def test_coords_to_connected_components():
    test_cases = [
        (set(), frozenset()),
        ({(0,0)}, frozenset([frozenset({(0,0)})])),
        ({(0,0), (1,0), (0,1)}, frozenset([frozenset({(0,0), (1,0), (0,1)})])),
        ({(0,0), (2,2)}, frozenset([frozenset({(0,0)}), frozenset({(2,2)})])),
        ({(0,0), (0,1), (0,2), (2,0), (2,1), (2,2)},
        frozenset([frozenset({(0,0), (0,1), (0,2)}), frozenset({(2,0), (2,1), (2,2)})])),
        # New test cases for 8-connectivity
        ({(0,0), (1,1)}, frozenset([frozenset({(0,0), (1,1)})])),
        ({(0,0), (1,1), (2,0), (0,2), (2,2)},
        frozenset([frozenset({(0,0), (1,1), (2,0), (0,2), (2,2)})])),
    ]
    for coords, expected in test_cases:
        result = coords_to_connected_components(coords)
        assert result == expected, f"Failed for coords={coords}: expected {expected}, got {result}"

def test_condition_by_color_couples():
    grid1 = [[0,1], [2,0]]
    expected1 = {
        (0,0): {(0,0), (1,1)},
        (1,0): {(0,0), (1,0), (1,1)},
        (1,1): {(1,0)},
        (2,0): {(0,0), (0,1), (1,1)},
        (2,1): {(0,1), (1,0)},
        (2,2): {(0, 1)}
    }
    result1 = condition_by_color_couples(grid1)
    assert result1 == expected1, f"Failed for grid1: expected {expected1}, got {result1}"

    grid2 = [[0,0], [0,0]]
    expected2 = {(0,0): {(0,0), (1,0), (0,1), (1,1)}}
    result2 = condition_by_color_couples(grid2)
    assert result2 == expected2, f"Failed for grid2: expected {expected2}, got {result2}"

    grid3 = [[0,1,2], [2,1,0]]
    expected3 = {
        (0,0): {(0,0), (2,1)},
        (1,0): {(0,0), (1,0), (1,1), (2,1)},
        (1,1): {(1,0), (1,1)},
        (2,0): {(0,0), (0,1), (2,0), (2,1)},
        (2,1): {(0,1), (1,0), (1,1), (2,0)},
        (2,2): {(2,0), (0,1)}
    }
    result3 = condition_by_color_couples(grid3)
    assert result3 == expected3, f"Failed for grid3: expected {expected3}, got {result3}"

def test_condition_by_colors():
    grid1 = [[0,1], [2,0]]
    result1 = condition_by_colors(grid1)
    assert result1[frozenset({0})] == {(0,0), (1,1)}
    assert result1[frozenset({1})] == {(1,0)}
    assert result1[frozenset({2})] == {(0,1)}
    assert result1[frozenset({0,1})] == {(0,0), (1,0), (1,1)}
    assert result1[frozenset({0,2})] == {(0,0), (0,1), (1,1)}
    assert result1[frozenset({1,2})] == {(1,0), (0,1)}
    assert result1[frozenset({0,1,2})] == {(0,0), (0,1), (1,0), (1,1)}
    assert len(result1) == 7  # 2^3 - 1
    print("Tests passed!")

def test_components_by_colors_to_grid_object_dag():
    # Define the input
    components_by_colors: dict[Colors, frozenset[Coords]] = {
        frozenset({0}): frozenset({frozenset({Coord(0,0)})}),
        frozenset({1}): frozenset({frozenset({Coord(1,1)})}),
        frozenset({2}): frozenset({frozenset({Coord(2,2)})}),
        frozenset({0,1}): frozenset({frozenset({Coord(0,0), Coord(1,1)})}),
        frozenset({0,2}): frozenset({frozenset({Coord(0,0), Coord(2,2)})}),
        frozenset({1,2}): frozenset({frozenset({Coord(1,1), Coord(2,2)})}),
        frozenset({0,1,2}): frozenset({frozenset({Coord(0,0), Coord(1,1), Coord(2,2)})})
    }

    # Compute the result
    result = components_by_colors_to_grid_object_dag(components_by_colors)

    # Define GridObjects for the expected output
    GO0 = GridObject(frozenset({0}), frozenset({Coord(0,0)}))
    GO1 = GridObject(frozenset({1}), frozenset({Coord(1,1)}))
    GO2 = GridObject(frozenset({2}), frozenset({Coord(2,2)}))
    GO01 = GridObject(frozenset({0,1}), frozenset({Coord(0,0), Coord(1,1)}))
    GO02 = GridObject(frozenset({0,2}), frozenset({Coord(0,0), Coord(2,2)}))
    GO12 = GridObject(frozenset({1,2}), frozenset({Coord(1,1), Coord(2,2)}))
    GO012 = GridObject(frozenset({0,1,2}), frozenset({Coord(0,0), Coord(1,1), Coord(2,2)}))

    # Define the expected DAG
    expected_graph = {
        GO012: {GO01, GO02, GO12},
        GO01: {GO0, GO1},
        GO02: {GO0, GO2},
        GO12: {GO1, GO2},
        GO0: set(),
        GO1: set(),
        GO2: set()
    }

    # Verify the result matches the expected graph
    assert set(result.keys()) == set(expected_graph.keys()), "DAG keys do not match expected keys"
    for key in expected_graph:
        assert result[key] == expected_graph[key], f"DAG mismatch for {key}: expected {expected_graph[key]}, got {result[key]}"


if __name__ == "__main__":
    test_coords_to_connected_components()
    test_condition_by_color_couples()
    test_condition_by_colors()
    test_components_by_colors_to_grid_object_dag()

    print("All tests passed!")
