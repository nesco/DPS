"""
Functionals for tree structures. It would work for graphs, if adapted by using `seen = set()`construct to avoid cycles
"""

from collections import deque
from typing import Callable, Generic, Iterator, Sequence, TypeVar

T = TypeVar("T")


# Test class for a rose tree
class RoseNode(Generic[T]):
    def __init__(self, value):
        self.value: T = value
        self.children: Sequence[T] = []


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
    A = RoseNode("A")
    B = RoseNode("B")
    C = RoseNode("C")
    D = RoseNode("D")
    E = RoseNode("E")
    F = RoseNode("F")
    A.children = [B, C]
    B.children = [D, E]
    C.children = [F]

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


if __name__ == "__main__":
    test_traversals()
    print("All tests passed successfully!")
