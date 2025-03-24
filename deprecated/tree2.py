"""
Implement a Rose Tree structure
"""

from collections import deque
from typing import TypeVar, Generic, Any, Callable
from dataclasses import dataclass, field

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

@dataclass(frozen=True)
class Leaf(Generic[U]):
    value: U

Node = 'RoseTree[T, U]' | Leaf[U]

@dataclass(frozen=True)
class RoseTree(Generic[T, U]):
    value: T
    children: list[Node] = field(default_factory=list)

    def __iter__(self):
        yield self.value
        for child in self.children:
            if isinstance(child, RoseTree):
                yield from child
            else:
                yield child.value

def next_layer(layer: list[Node]) -> list[Node]:
    return [child for node in layer if isinstance(node, RoseTree) for child in node.children]


def leaves(tree: Node) -> list[Leaf[U]]:
    """Imperative BFS algorithm to get all leaves of a rose tree"""

    queue = deque([tree])
    leaf_list = []

    while queue:
        head = queue.popleft()
        match head:
            case Leaf():
                leaf_list.append(head)
            case RoseTree():
                queue.extend(head.children)

    return leaf_list

def depth(tree: Node) -> int:
    """Return the height of a rose tree"""

    queue = [tree]
    depth = -1

    while queue:
        # Increment the depth
        depth += 1
        # Fetching the next layer
        queue = next_layer(queue)

    return depth

def values_bfs(tree: Node) -> list[T | U]:
    """Return the values of a Rose Tree in a breadth-first order"""

    queue = [tree]
    values = []

    while queue:
        # Adding the values of the current layers to the value list
        values.extend([node.value for node in queue])
        # Fetching the next layer
        queue = next_layer(queue)

    return values

def values_dfs(tree: Node) -> list[T | U]:
    """Return the values of a Rose Tree in a depth-first order"""
    queue = [tree]
    values = []

    while queue:
        head = queue.pop()
        values.append(head.value)
        if isinstance(head, RoseTree):
            queue += reversed(head.children)

    return values

def count(tree: Node) -> int:
    """Return the node count of a Rose Tree"""

    queue = [tree]
    count = 0

    while queue:
        # Counting the number of node of the current layer
        count += len(queue)
        # Fetching the next layer
        queue =  next_layer(queue)

    return count

def map_node(tree: Node, f: Callable[[T | U], Any]) -> Node:
    """Apply a function to every node value in a rose tree (both leaf and non-leaf nodes).

    Args:
        tree: The rose tree to transform
        f: The function to apply to each node value

    Returns:
        A new rose tree with the function applied to all node values
    """
    match tree:
        case Leaf(value):
           return Leaf(f(value))

        case RoseTree(value, children):
            return RoseTree(value=f(value), children=[map_node(child, f) for child in children])

def map_leaf(tree: Node, f: Callable[[U], Any]) -> Node:
    """Apply a function only to leaf node values in a rose tree, preserving non-leaf node values.

    Args:
        tree: The rose tree to transform
        f: The function to apply to leaf node values

    Returns:
        A new rose tree with the function applied only to leaf values
    """
    match tree:
        case Leaf(value):
          return Leaf(f(value))

        case RoseTree(value, children):
            return RoseTree(value=value, children=[map_node(child, f) for child in children])

def map_tree(tree: Node, f: Callable[[U], Any]) -> Node:
    """Apply a function only to non-leaf (tree) node values in a rose tree, preserving leaf values.

    Args:
        tree: The rose tree to transform
        f: The function to apply to non-leaf node values

    Returns:
        A new rose tree with the function applied only to non-leaf values
    """
    match tree:
        case Leaf(value):
          return Leaf(value)

        case RoseTree(value, children):
            return RoseTree(value=f(value), children=[map_node(child, f) for child in children])

# Test functions
def test_traversal_functions():
    # Test 1: Single node tree.
    tree1 = RoseTree(1, [])
    expected_bfs_tree1 = [1]
    expected_dfs_tree1 = [1]
    assert values_bfs(tree1) == expected_bfs_tree1, f"values_bfs failed for tree1. Expected {expected_bfs_tree1}, got {values_bfs(tree1)}"
    assert values_dfs(tree1) == expected_dfs_tree1, f"values_dfs failed for tree1. Expected {expected_dfs_tree1}, got {values_dfs(tree1)}"

    # Test 2: Tree with only leaves.
    tree2 = RoseTree(1, [Leaf(2), Leaf(3), Leaf(4)])
    expected_bfs_tree2 = [1, 2, 3, 4]
    expected_dfs_tree2 = [1, 2, 3, 4]
    assert values_bfs(tree2) == expected_bfs_tree2, f"values_bfs failed for tree2. Expected {expected_bfs_tree2}, got {values_bfs(tree2)}"
    assert values_dfs(tree2) == expected_dfs_tree2, f"values_dfs failed for tree2. Expected {expected_dfs_tree2}, got {values_dfs(tree2)}"

    # Test 3: More complex tree.
    tree3 = RoseTree(1, [
        RoseTree(2, [Leaf(3), Leaf(4)]),
        Leaf(5),
        RoseTree(6, [
            RoseTree(7, [Leaf(8)]),
            Leaf(9)
        ])
    ])
    # Expected BFS order:
    # Level 0: 1
    # Level 1: 2, 5, 6
    # Level 2: 3, 4, 7, 9
    # Level 3: 8
    expected_bfs_tree3 = [1, 2, 5, 6, 3, 4, 7, 9, 8]

    # Expected DFS (pre-order) order:
    # 1, then 2, 3, 4, then 5, then 6, then 7, 8, then 9
    expected_dfs_tree3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert values_bfs(tree3) == expected_bfs_tree3, f"values_bfs failed for tree3. Expected {expected_bfs_tree3}, got {values_bfs(tree3)}"
    assert values_dfs(tree3) == expected_dfs_tree3, f"values_dfs failed for tree3. Expected {expected_dfs_tree3}, got {values_dfs(tree3)}"

    print("All traversal function tests passed!")

def test_map_functions():
    # Test map_node
    tree1 = RoseTree(1, [Leaf(2), RoseTree(3, [Leaf(4)])])
    mapped_tree1 = map_node(tree1, lambda x: x * 2)
    assert list(mapped_tree1) == [2, 4, 6, 8], "map_node failed"

    # Test map_leaf
    tree2 = RoseTree(1, [Leaf(2), RoseTree(3, [Leaf(4)])])
    mapped_tree2 = map_leaf(tree2, lambda x: x * 2)
    assert list(mapped_tree2) == [1, 4, 3, 8], "map_leaf failed"

    # Test map_tree
    tree3 = RoseTree(1, [Leaf(2), RoseTree(3, [Leaf(4)])])
    mapped_tree3 = map_tree(tree3, lambda x: x * 2)
    assert list(mapped_tree3) == [2, 2, 6, 4], "map_tree failed"

    print("All map function tests passed!")

if __name__ == "__main__":
    # Create a simple tree with only a root
    tree1 = RoseTree(1, [])
    assert list(tree1) == [1]

    # Create a tree with some leaves
    tree2 = RoseTree(1, [Leaf(2), Leaf(3), Leaf(4)])
    assert list(tree2) == [1, 2, 3, 4]

    # Create a more complex tree with nested RoseTrees and Leaves
    tree3 = RoseTree(1, [
        RoseTree(2, [Leaf(3), Leaf(4)]),
        Leaf(5),
        RoseTree(6, [
            RoseTree(7, [Leaf(8)]),
            Leaf(9)
        ])
    ])
    assert list(tree3) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Test with different types
    tree4 = RoseTree("root", [
        Leaf("leaf1"),
        RoseTree("branch1", [Leaf("leaf2")]),
        Leaf("leaf3")
    ])
    assert list(tree4) == ["root", "leaf1", "branch1", "leaf2", "leaf3"]

    # Additional tests for leaves function
    tree5 = RoseTree(1, [Leaf(2), RoseTree(3, [Leaf(4), Leaf(5)])])
    assert leaves(tree5) == [Leaf(2), Leaf(4), Leaf(5)]

    tree6 = RoseTree("root", [Leaf("a"), RoseTree("b", [Leaf("c")])])
    assert leaves(tree6) == [Leaf("a"), Leaf("c")]

    # Tests for depth function
    # Single node tree has depth 0
    tree7 = RoseTree(1, [])
    assert depth(tree7) == 0

    # Tree with only leaves has depth 1
    tree8 = RoseTree(1, [Leaf(2), Leaf(3)])
    assert depth(tree8) == 1

    # Tree with nested structure
    tree9 = RoseTree(1, [
        RoseTree(2, [Leaf(3)]),
        RoseTree(4, [
            RoseTree(5, [Leaf(6)])
        ])
    ])
    assert depth(tree9) == 3

    # Tests for count function
    # Single node tree has count 1
    assert count(tree7) == 1

    # Tree with only leaves counts root and leaves
    assert count(tree8) == 3

    # Tree with nested structure counts all RoseTree nodes
    assert count(tree9) == 6

    test_traversal_functions()
    test_map_functions()

    print("All tests passed!")
