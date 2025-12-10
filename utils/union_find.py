"""
Union-Find (Disjoint Set Union) data structure.

Efficient data structure for tracking disjoint sets with:
- find(x): Which set contains x? - O(α(n)) amortized
- union(x, y): Merge sets containing x and y - O(α(n)) amortized
- connected(x, y): Are x and y in the same set? - O(α(n)) amortized

Where α(n) is the inverse Ackermann function (effectively constant ≤ 4).

Currently unused but available for future optimizations
(e.g., incremental connected component maintenance).
"""

from typing import Generic, TypeVar

Element = TypeVar("Element")


class UnionFind(Generic[Element]):
    """
    Union-Find with path compression and union by rank.

    Elements are lazily initialized on first use.

    Example:
        >>> uf = UnionFind[int]()
        >>> uf.union(1, 2)
        >>> uf.union(2, 3)
        >>> uf.connected(1, 3)
        True
        >>> uf.connected(1, 4)
        False
    """

    def __init__(self) -> None:
        self._parent: dict[Element, Element] = {}
        self._rank: dict[Element, int] = {}

    def _ensure_exists(self, element: Element) -> None:
        """Initialize element as its own set if not seen before."""
        if element not in self._parent:
            self._parent[element] = element
            self._rank[element] = 0

    def find(self, element: Element) -> Element:
        """
        Find the representative (root) of the set containing element.

        Uses path compression: flattens the tree by pointing all nodes
        along the path directly to the root.
        """
        self._ensure_exists(element)

        # Find root
        root = element
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression: point all nodes to root
        current = element
        while self._parent[current] != root:
            next_node = self._parent[current]
            self._parent[current] = root
            current = next_node

        return root

    def union(self, x: Element, y: Element) -> Element:
        """
        Merge the sets containing x and y.

        Uses union by rank: attaches the shorter tree under the taller one
        to keep trees balanced.

        Returns the representative of the merged set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return root_x

        # Attach smaller tree under larger tree
        if self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
            return root_y
        elif self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
            return root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1
            return root_x

    def connected(self, x: Element, y: Element) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def get_all_sets(self) -> dict[Element, set[Element]]:
        """
        Get all disjoint sets as a dictionary.

        Returns:
            Mapping from each set's representative to its members.
        """
        sets: dict[Element, set[Element]] = {}
        for element in self._parent:
            root = self.find(element)
            if root not in sets:
                sets[root] = set()
            sets[root].add(element)
        return sets
