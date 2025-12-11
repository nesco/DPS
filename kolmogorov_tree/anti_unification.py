"""
Anti-unification (Least General Generalization) for Kolmogorov Trees.

Anti-unification finds the most specific pattern that generalizes two trees.
Unlike hardcoded template extraction, this discovers patterns from data.

Example:
    T1 = RootNode(ProductNode(A, B), pos1, color1)
    T2 = RootNode(ProductNode(A, C), pos2, color1)

    anti_unify(T1, T2) → RootNode(ProductNode(A, Var0), Var1, color1)
                         with bindings {0: (B, C), 1: (pos1, pos2)}
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Sequence

from kolmogorov_tree.nodes import (
    KNode,
    NestedNode,
    PrimitiveNode,
    ProductNode,
    RectNode,
    RepeatNode,
    RootNode,
    SumNode,
    SymbolNode,
    VariableNode,
)
from kolmogorov_tree.predicates import is_abstraction
from kolmogorov_tree.primitives import BitLength, IndexValue, T, VariableValue
from kolmogorov_tree.types import BitLengthAware


@dataclass
class AntiUnificationResult:
    """Result of anti-unifying two trees."""

    pattern: KNode  # The generalized pattern
    bindings1: dict[int, BitLengthAware]  # Variable bindings for tree 1
    bindings2: dict[int, BitLengthAware]  # Variable bindings for tree 2

    def parameters_for(self, tree_idx: int) -> tuple[BitLengthAware, ...]:
        """Get ordered parameters for a tree."""
        bindings = self.bindings1 if tree_idx == 0 else self.bindings2
        if not bindings:
            return ()
        max_var = max(bindings.keys())
        return tuple(bindings.get(i) for i in range(max_var + 1) if i in bindings)


class _AntiUnifier:
    """Stateful anti-unification with variable counter."""

    def __init__(self):
        self.var_counter = 0
        self.bindings1: dict[int, BitLengthAware] = {}
        self.bindings2: dict[int, BitLengthAware] = {}

    def fresh_var(self, val1: BitLengthAware, val2: BitLengthAware) -> VariableNode:
        """Create a fresh variable and record bindings."""
        var_idx = self.var_counter
        self.var_counter += 1
        self.bindings1[var_idx] = val1
        self.bindings2[var_idx] = val2
        return VariableNode(VariableValue(var_idx))

    def anti_unify(self, t1: BitLengthAware, t2: BitLengthAware) -> BitLengthAware:
        """Compute the least general generalization of t1 and t2."""
        # Identical values need no generalization
        if t1 == t2:
            return t1

        # Both must be KNodes for structural comparison
        if not isinstance(t1, KNode) or not isinstance(t2, KNode):
            return self.fresh_var(t1, t2)

        # Different types → generalize to variable
        if type(t1) != type(t2):
            return self.fresh_var(t1, t2)

        # Same type, recurse based on node structure
        match t1:
            case PrimitiveNode(value1):
                t2 = t2  # type: PrimitiveNode
                if value1 == t2.value:
                    return t1
                return self.fresh_var(t1, t2)

            case VariableNode():
                # Variables are already abstract
                return self.fresh_var(t1, t2)

            case ProductNode(children1):
                t2 = t2  # type: ProductNode
                children2 = t2.children
                if len(children1) != len(children2):
                    return self.fresh_var(t1, t2)
                new_children = tuple(
                    self.anti_unify(c1, c2) for c1, c2 in zip(children1, children2)
                )
                return ProductNode(new_children)

            case SumNode(children1):
                t2 = t2  # type: SumNode
                children2 = t2.children
                if len(children1) != len(children2):
                    return self.fresh_var(t1, t2)
                # For unordered sets, try to find best alignment
                result = self._anti_unify_sets(list(children1), list(children2))
                if result is None:
                    return self.fresh_var(t1, t2)
                return SumNode(frozenset(result))

            case RepeatNode(node1, count1):
                t2 = t2  # type: RepeatNode
                new_node = self.anti_unify(node1, t2.node)
                new_count = self.anti_unify(count1, t2.count)
                return RepeatNode(new_node, new_count)

            case NestedNode(index1, node1, count1):
                t2 = t2  # type: NestedNode
                if index1 != t2.index:
                    return self.fresh_var(t1, t2)
                new_node = self.anti_unify(node1, t2.node)
                new_count = self.anti_unify(count1, t2.count)
                return NestedNode(index1, new_node, new_count)

            case SymbolNode(index1, params1):
                t2 = t2  # type: SymbolNode
                if index1 != t2.index or len(params1) != len(t2.parameters):
                    return self.fresh_var(t1, t2)
                new_params = tuple(
                    self.anti_unify(p1, p2) for p1, p2 in zip(params1, t2.parameters)
                )
                return SymbolNode(index1, new_params)

            case RectNode(h1, w1):
                t2 = t2  # type: RectNode
                new_h = self.anti_unify(h1, t2.height)
                new_w = self.anti_unify(w1, t2.width)
                return RectNode(new_h, new_w)

            case RootNode(node1, pos1, colors1):
                t2 = t2  # type: RootNode
                new_node = self.anti_unify(node1, t2.node)
                new_pos = self.anti_unify(pos1, t2.position)
                new_colors = self.anti_unify(colors1, t2.colors)
                return RootNode(new_node, new_pos, new_colors)

            case _:
                return self.fresh_var(t1, t2)

    def _anti_unify_sets(
        self, set1: list[KNode], set2: list[KNode]
    ) -> list[KNode] | None:
        """Anti-unify unordered sets by finding best alignment."""
        if len(set1) != len(set2):
            return None

        n = len(set1)
        if n == 0:
            return []

        # Greedy matching: pair elements that are most similar
        # Similarity = number of variables needed (fewer is better)
        used2 = [False] * n
        result = []

        for i, elem1 in enumerate(set1):
            best_j = -1
            best_vars = float("inf")
            best_result = None

            for j, elem2 in enumerate(set2):
                if used2[j]:
                    continue

                # Try anti-unifying this pair
                test_unifier = _AntiUnifier()
                test_unifier.var_counter = self.var_counter
                test_result = test_unifier.anti_unify(elem1, elem2)
                num_vars = test_unifier.var_counter - self.var_counter

                if num_vars < best_vars:
                    best_vars = num_vars
                    best_j = j
                    best_result = (test_result, test_unifier)

            if best_j >= 0 and best_result is not None:
                used2[best_j] = True
                unified, unifier = best_result
                # Merge the bindings
                for k, v in unifier.bindings1.items():
                    if k >= self.var_counter:
                        self.bindings1[k] = v
                for k, v in unifier.bindings2.items():
                    if k >= self.var_counter:
                        self.bindings2[k] = v
                self.var_counter = unifier.var_counter
                result.append(unified)
            else:
                return None

        return result


def anti_unify(t1: KNode[T], t2: KNode[T]) -> AntiUnificationResult:
    """
    Compute the least general generalization of two trees.

    Returns a pattern with variables where t1 and t2 differ,
    along with the bindings that recover each original tree.
    """
    unifier = _AntiUnifier()
    pattern = unifier.anti_unify(t1, t2)
    return AntiUnificationResult(
        pattern=pattern,
        bindings1=unifier.bindings1,
        bindings2=unifier.bindings2,
    )


def discover_templates_pairwise(
    subtrees: Sequence[KNode[T]],
    min_occurrences: int = 2,
    max_variables: int = 3,
) -> dict[KNode, list[KNode]]:
    """
    Discover templates by pairwise anti-unification of subtrees.

    Returns a dict mapping template patterns to the subtrees they match.
    Only returns templates that:
    - Match at least min_occurrences subtrees
    - Have at most max_variables variables
    - Are non-trivial (not just a single variable)
    """
    # Deduplicate subtrees
    unique_subtrees = list(set(subtrees))
    n = len(unique_subtrees)

    if n < min_occurrences:
        return {}

    # Count occurrences of each subtree
    subtree_counts = Counter(subtrees)

    # Collect templates and their matches
    template_matches: dict[KNode, list[KNode]] = {}

    # Add exact matches first
    for subtree, count in subtree_counts.items():
        if count >= min_occurrences and not is_abstraction(subtree):
            template_matches[subtree] = [subtree] * count

    # Pairwise anti-unification for non-identical subtrees
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = unique_subtrees[i], unique_subtrees[j]

            # Skip if either already contains variables
            if is_abstraction(t1) or is_abstraction(t2):
                continue

            result = anti_unify(t1, t2)

            # Skip trivial patterns (single variable)
            if isinstance(result.pattern, VariableNode):
                continue

            # Skip patterns with too many variables
            num_vars = len(result.bindings1)
            if num_vars > max_variables or num_vars == 0:
                continue

            # Add to matches
            pattern = result.pattern
            if pattern not in template_matches:
                template_matches[pattern] = []

            # Add both original subtrees as matches (with their counts)
            for _ in range(subtree_counts[t1]):
                if t1 not in template_matches[pattern]:
                    template_matches[pattern].append(t1)
            for _ in range(subtree_counts[t2]):
                if t2 not in template_matches[pattern]:
                    template_matches[pattern].append(t2)

    # Filter to patterns with enough occurrences
    return {
        pat: matches
        for pat, matches in template_matches.items()
        if len(matches) >= min_occurrences
    }


@cache
def extract_template_anti_unify(knode: KNode[T]) -> tuple[tuple[KNode[T], tuple], ...]:
    """
    Alternative to extract_template using self-anti-unification.

    For a single node, we can't do pairwise comparison, so we
    generate templates by abstracting "abstractable" parts.
    This is a hybrid approach that uses anti-unification logic
    but applies it systematically to a single node.
    """
    if is_abstraction(knode):
        return ()

    abstractions: list[tuple[KNode[T], tuple]] = []

    # Get all abstractable positions in the node
    positions = _get_abstractable_positions(knode)

    # Generate templates by abstracting subsets of positions
    for pos in positions:
        template, params = _abstract_position(knode, pos)
        if template != knode:  # Only add if actually abstracted
            abstractions.append((template, params))

    return tuple(abstractions)


def _get_abstractable_positions(knode: KNode) -> list[tuple[str, ...]]:
    """Get paths to abstractable (non-variable, non-trivial) positions."""
    positions = []

    match knode:
        case ProductNode(children) if len(children) > 0:
            # Abstract repeated children
            child_counts = Counter(children)
            for child, count in child_counts.items():
                if count >= 1 and not is_abstraction(child):
                    positions.append(("children", child))

        case SumNode(children) if len(children) > 0:
            for child in children:
                if not is_abstraction(child):
                    positions.append(("children", child))

        case RepeatNode(node, count):
            if not is_abstraction(node):
                positions.append(("node",))
            if not isinstance(count, VariableNode):
                positions.append(("count",))

        case RootNode(node, position, colors):
            if not is_abstraction(node):
                positions.append(("node",))
            if not isinstance(position, VariableNode):
                positions.append(("position",))
            if not isinstance(colors, VariableNode):
                positions.append(("colors",))

        case RectNode(height, width):
            if not isinstance(height, VariableNode):
                positions.append(("height",))
            if not isinstance(width, VariableNode):
                positions.append(("width",))
            if height == width and not isinstance(height, VariableNode):
                positions.append(("both",))

        case NestedNode(_, node, count):
            if not is_abstraction(node):
                positions.append(("node",))
            if not isinstance(count, VariableNode):
                positions.append(("count",))

    return positions


def _abstract_position(
    knode: KNode, position: tuple
) -> tuple[KNode, tuple[BitLengthAware, ...]]:
    """Abstract a specific position in a node."""
    var0 = VariableNode(VariableValue(0))
    var1 = VariableNode(VariableValue(1))

    match (knode, position):
        case (ProductNode(children), ("children", target)):
            new_children = tuple(var0 if c == target else c for c in children)
            return ProductNode(new_children), (target,)

        case (SumNode(children), ("children", target)):
            new_children = frozenset(var0 if c == target else c for c in children)
            return SumNode(new_children), (target,)

        case (RepeatNode(node, count), ("node",)):
            return RepeatNode(var0, count), (node,)

        case (RepeatNode(node, count), ("count",)):
            return RepeatNode(node, var0), (count,)

        case (RootNode(node, pos, colors), ("node",)):
            return RootNode(var0, pos, colors), (node,)

        case (RootNode(node, pos, colors), ("position",)):
            return RootNode(node, var0, colors), (pos,)

        case (RootNode(node, pos, colors), ("colors",)):
            return RootNode(node, pos, var0), (colors,)

        case (RectNode(h, w), ("height",)):
            return RectNode(var0, w), (h,)

        case (RectNode(h, w), ("width",)):
            return RectNode(h, var0), (w,)

        case (RectNode(h, w), ("both",)):
            return RectNode(var0, var0), (h,)

        case (NestedNode(idx, node, count), ("node",)):
            return NestedNode(idx, var0, count), (node,)

        case (NestedNode(idx, node, count), ("count",)):
            return NestedNode(idx, node, var0), (count,)

        case _:
            return knode, ()


__all__ = [
    "AntiUnificationResult",
    "anti_unify",
    "discover_templates_pairwise",
    "extract_template_anti_unify",
]
