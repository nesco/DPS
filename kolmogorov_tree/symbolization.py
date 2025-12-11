"""
Symbolization for Kolmogorov Tree.

Uses DAG-based pattern discovery for efficient symbolization.

Functions:
    Discovery:
        find_symbol_candidates(trees)   - Find frequent patterns ranked by bit savings

    Symbolization:
        symbolize_pattern(trees, symbols, pattern) - Replace pattern with SymbolNode
        symbolize(trees, symbols)                  - DAG-based greedy symbolization
        full_symbolization(trees)                  - Complete pipeline with nesting

    Symbol Table Operations:
        merge_symbol_tables(tables)        - Unify multiple symbol tables
        symbolize_together(trees, tables)  - Merge tables and re-symbolize

    Unsymbolization:
        unsymbolize(node, table)     - Expand all symbols in a node
        unsymbolize_all(trees, table) - Expand symbols in multiple trees

    Refactoring:
        factor_by_existing_symbols(tree, symbols) - Match tree against symbol table
        remap_symbol_indices(tree, mapping)       - Update SymbolNode indices
"""

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from utils.algorithms.dag import topological_sort

from kolmogorov_tree.matching import abstract_node, node_to_symbolized_node
from kolmogorov_tree.nodes import (
    KNode,
    SymbolNode,
)
from kolmogorov_tree.predicates import (
    arity,
    contained_symbols,
    is_abstraction,
)
from kolmogorov_tree.primitives import BitLength, IndexValue, T
from kolmogorov_tree.substitution import (
    expand_all_nested_nodes,
    extract_nested_patterns,
    resolve_symbols,
)
from kolmogorov_tree.templates import extract_template
from kolmogorov_tree.transformations import postmap
from kolmogorov_tree.traversal import breadth_first_preorder_knode
from kolmogorov_tree.anti_unification import discover_templates_pairwise


@dataclass
class _PatternInfo:
    """Information about a pattern for bit-gain calculation."""

    pattern: KNode
    count: int
    matches: list[KNode]
    avg_param_bits: float = 0.0

    def bit_gain(self) -> float:
        """Calculate bit savings from symbolizing this pattern."""
        current_len = sum(m.bit_length() for m in self.matches)
        symbol_overhead = BitLength.NODE_TYPE + BitLength.INDEX + int(self.avg_param_bits)
        return current_len - (self.count * symbol_overhead + self.pattern.bit_length())


@dataclass
class _SymbolizationDAG:
    """DAG for efficient pattern discovery and symbolization."""

    pattern_info: dict[KNode, _PatternInfo] = field(default_factory=dict)
    subtree_templates: dict[KNode, list[tuple[KNode, tuple]]] = field(default_factory=dict)
    # Per-tree subtree lists for incremental updates
    _tree_subtrees: list[list[KNode]] = field(default_factory=list)

    @classmethod
    def from_trees(cls, trees: Sequence[KNode[T]]) -> "_SymbolizationDAG":
        """Build DAG from trees in a single pass."""
        dag = cls()
        dag._build_from_trees(trees)
        return dag

    def _build_from_trees(self, trees: Sequence[KNode[T]]) -> None:
        """Build pattern info from trees."""
        self._tree_subtrees = []
        pattern_counter: Counter[KNode[T]] = Counter()
        pattern_matches: defaultdict[KNode[T], list[KNode[T]]] = defaultdict(list)

        for tree in trees:
            tree_subtrees = list(breadth_first_preorder_knode(tree))
            self._tree_subtrees.append(tree_subtrees)

            for subtree in tree_subtrees:
                pattern_counter[subtree] += 1
                pattern_matches[subtree].append(subtree)

                if subtree not in self.subtree_templates:
                    self.subtree_templates[subtree] = list(extract_template(subtree))

                for template, _ in self.subtree_templates[subtree]:
                    pattern_counter[template] += 1
                    pattern_matches[template].append(subtree)

        self._build_pattern_info(pattern_counter, pattern_matches)

    def _build_pattern_info(
        self,
        pattern_counter: Counter[KNode[T]],
        pattern_matches: defaultdict[KNode[T], list[KNode[T]]],
    ) -> None:
        """Build PatternInfo from counters."""
        self.pattern_info.clear()
        for pattern, count in pattern_counter.items():
            matches = pattern_matches[pattern]

            avg_param_bits = 0.0
            if is_abstraction(pattern) and matches:
                total_param_bits = 0
                for match in matches:
                    for tmpl, params in self.subtree_templates.get(match, []):
                        if tmpl == pattern:
                            total_param_bits += sum(p.bit_length() for p in params)
                            break
                avg_param_bits = total_param_bits / len(matches)

            self.pattern_info[pattern] = _PatternInfo(
                pattern=pattern,
                count=count,
                matches=matches,
                avg_param_bits=avg_param_bits,
            )

    def update_trees(
        self,
        old_trees: Sequence[KNode[T]],
        new_trees: Sequence[KNode[T]],
    ) -> None:
        """
        Incrementally update DAG when trees change.

        Only reprocesses trees that actually changed (different hash).
        Reuses template cache for unchanged subtrees.
        """
        if len(old_trees) != len(new_trees):
            # Structure changed, full rebuild
            self._build_from_trees(new_trees)
            return

        # Find which trees changed
        changed_indices = [
            i for i, (old, new) in enumerate(zip(old_trees, new_trees))
            if old != new
        ]

        if not changed_indices:
            return

        # If most trees changed, full rebuild is faster
        if len(changed_indices) > len(old_trees) // 2:
            self._build_from_trees(new_trees)
            return

        # Incremental update: remove old counts, add new counts
        pattern_counter: Counter[KNode[T]] = Counter()
        pattern_matches: defaultdict[KNode[T], list[KNode[T]]] = defaultdict(list)

        # Rebuild from existing pattern_info
        for pattern, info in self.pattern_info.items():
            pattern_counter[pattern] = info.count
            pattern_matches[pattern] = list(info.matches)

        # Remove counts from changed trees
        for idx in changed_indices:
            if idx < len(self._tree_subtrees):
                for subtree in self._tree_subtrees[idx]:
                    pattern_counter[subtree] -= 1
                    if subtree in pattern_matches:
                        try:
                            pattern_matches[subtree].remove(subtree)
                        except ValueError:
                            pass

                    for template, _ in self.subtree_templates.get(subtree, []):
                        pattern_counter[template] -= 1
                        if template in pattern_matches:
                            try:
                                pattern_matches[template].remove(subtree)
                            except ValueError:
                                pass

        # Add counts from new trees
        for idx in changed_indices:
            new_tree = new_trees[idx]
            tree_subtrees = list(breadth_first_preorder_knode(new_tree))

            if idx < len(self._tree_subtrees):
                self._tree_subtrees[idx] = tree_subtrees
            else:
                self._tree_subtrees.append(tree_subtrees)

            for subtree in tree_subtrees:
                pattern_counter[subtree] += 1
                pattern_matches[subtree].append(subtree)

                if subtree not in self.subtree_templates:
                    self.subtree_templates[subtree] = list(extract_template(subtree))

                for template, _ in self.subtree_templates[subtree]:
                    pattern_counter[template] += 1
                    pattern_matches[template].append(subtree)

        # Clean up zero counts
        pattern_counter = Counter({k: v for k, v in pattern_counter.items() if v > 0})
        pattern_matches = defaultdict(list, {k: v for k, v in pattern_matches.items() if v})

        self._build_pattern_info(pattern_counter, pattern_matches)

    def get_best_candidates(
        self,
        min_occurrences: int = 2,
        max_patterns: int = 10,
    ) -> list[KNode]:
        """Get patterns ranked by bit gain."""
        candidates = []

        for pattern, info in self.pattern_info.items():
            if info.count < min_occurrences:
                continue

            if is_abstraction(pattern) and len(info.matches) < min_occurrences:
                continue

            gain = info.bit_gain()
            if gain > 0:
                candidates.append((pattern, gain))

        candidates.sort(key=lambda x: (-x[1], -x[0].bit_length()))
        return [p for p, _ in candidates[:max_patterns]]

    def enhance_with_anti_unification(
        self,
        max_subtrees: int = 100,
        min_occurrences: int = 3,
        max_variables: int = 2,
    ) -> None:
        """
        Discover additional templates using pairwise anti-unification.

        This finds patterns that span different subtree structures which
        hardcoded template extraction might miss. O(n^2) so uses sampling.

        Only adds patterns that:
        - Are not already known from hardcoded extraction
        - Have significant bit savings (high pattern cost, low variable count)
        - Occur frequently enough to justify the symbol overhead
        """
        from kolmogorov_tree.matching import matches as pattern_matches

        # Collect unique subtrees that haven't been abstracted yet
        concrete_subtrees = [
            s for s in self.subtree_templates.keys()
            if not is_abstraction(s)
        ]

        # Sample if too many - prefer larger subtrees (more compression potential)
        if len(concrete_subtrees) > max_subtrees:
            size_sorted = sorted(
                concrete_subtrees,
                key=lambda s: s.bit_length(),
                reverse=True,
            )
            concrete_subtrees = size_sorted[:max_subtrees]

        # Discover templates via anti-unification
        discovered = discover_templates_pairwise(
            concrete_subtrees,
            min_occurrences=min_occurrences,
            max_variables=max_variables,
        )

        # Filter and add only beneficial templates
        for template, template_matches in discovered.items():
            if template in self.pattern_info:
                continue  # Already known

            # Check if this template is already covered by hardcoded templates
            already_covered = False
            for existing_template in list(self.pattern_info.keys()):
                if is_abstraction(existing_template) and pattern_matches(existing_template, template):
                    already_covered = True
                    break
            if already_covered:
                continue

            # Calculate actual average parameter bits from matches
            total_param_bits = 0
            valid_matches = []
            for match in template_matches:
                bindings = pattern_matches(template, match)
                if bindings:
                    param_bits = sum(v.bit_length() for v in bindings.values())
                    total_param_bits += param_bits
                    valid_matches.append(match)

            if len(valid_matches) < min_occurrences:
                continue

            avg_param_bits = total_param_bits / len(valid_matches)

            # Only add if estimated bit gain is positive
            template_bits = template.bit_length()
            symbol_overhead = BitLength.NODE_TYPE + BitLength.INDEX + avg_param_bits
            match_bits = sum(m.bit_length() for m in valid_matches)
            estimated_gain = match_bits - (len(valid_matches) * symbol_overhead + template_bits)

            if estimated_gain <= 0:
                continue

            # Register this template for each matching subtree
            for match in valid_matches:
                if match in self.subtree_templates:
                    bindings = pattern_matches(template, match)
                    if bindings:
                        params = tuple(bindings.get(i) for i in sorted(bindings.keys()))
                        existing_templates = [t for t, _ in self.subtree_templates[match]]
                        if template not in existing_templates:
                            self.subtree_templates[match].append((template, params))

            self.pattern_info[template] = _PatternInfo(
                pattern=template,
                count=len(valid_matches),
                matches=list(valid_matches),
                avg_param_bits=avg_param_bits,
            )


def find_symbol_candidates(
    trees: Sequence[KNode[T]],
    min_occurrences: int = 2,
    max_patterns: int = 10,
) -> list[KNode]:
    """
    Finds frequent subtrees ranked by bit-length savings.

    Uses DAG-based pattern discovery for efficiency.
    Returns top patterns where symbolization reduces total bit length.
    Considers both concrete matches and abstracted patterns (with variables).
    """
    dag = _SymbolizationDAG.from_trees(trees)
    return dag.get_best_candidates(min_occurrences, max_patterns)


def symbolize_pattern(
    trees: Sequence[KNode[T]],
    symbols: Sequence[KNode[T]],
    new_symbol: KNode[T],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """Replaces all occurrences of new_symbol pattern with a SymbolNode reference."""
    index = IndexValue(len(symbols))
    trees = tuple(node_to_symbolized_node(index, new_symbol, tree) for tree in trees)
    symbols = tuple(
        node_to_symbolized_node(index, new_symbol, tree) for tree in symbols
    ) + (new_symbol,)
    return trees, symbols


def symbolize(
    trees: tuple[KNode[T], ...],
    symbols: tuple[KNode[T], ...],
    max_iterations: int = 100,
    use_anti_unification: bool = False,
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    DAG-based greedy symbolization with incremental updates.

    Builds a DAG to efficiently discover patterns, then iteratively
    symbolizes the highest-gain pattern until no savings remain.
    Uses incremental DAG updates to avoid full rebuilds when few trees change.

    Args:
        trees: Trees to symbolize.
        symbols: Existing symbol table.
        max_iterations: Safety limit on iterations.
        use_anti_unification: If True, enhance template discovery with
            pairwise anti-unification. Finds more patterns but O(n^2).
    """
    # Initial DAG build
    all_nodes = trees + symbols
    dag = _SymbolizationDAG.from_trees(all_nodes)

    # Optionally enhance with anti-unification
    if use_anti_unification:
        dag.enhance_with_anti_unification()

    for _ in range(max_iterations):
        candidates = dag.get_best_candidates()
        if not candidates:
            break

        new_symbol = candidates[0]
        index = IndexValue(len(symbols))

        # Apply symbolization
        old_trees = trees
        old_symbols = symbols
        trees = tuple(node_to_symbolized_node(index, new_symbol, tree) for tree in trees)
        symbols = tuple(
            node_to_symbolized_node(index, new_symbol, s) for s in symbols
        ) + (new_symbol,)

        # Incremental DAG update
        old_all = old_trees + old_symbols
        new_all = trees + symbols
        dag.update_trees(old_all, new_all)

    return trees, symbols


def factor_by_existing_symbols(
    tree: KNode[T], symbols: tuple[KNode[T], ...]
) -> KNode[T]:
    """Replaces subtrees matching existing symbols with SymbolNode references."""

    def factor_node(node: KNode[T]) -> KNode[T]:
        for i, symbol in enumerate(symbols):
            abstracted = abstract_node(IndexValue(i), symbol, node)
            if abstracted != node:
                return abstracted
        return node

    return postmap(tree, factor_node, factorize=True)


def remap_symbol_indices(tree: KNode[T], mapping: list[int], tree_idx: int) -> KNode[T]:
    """Updates SymbolNode indices according to the provided mapping."""

    def update_node(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters)
        return node

    return postmap(tree, update_node, factorize=False)


def remap_sub_symbols(
    symbol: KNode[T], mapping: list[int], original_table: tuple[KNode[T], ...]
) -> KNode[T]:
    """Remaps SymbolNode indices within a symbol definition."""

    def update_index(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters)
        return node

    return postmap(symbol, update_index, factorize=False)


def merge_symbol_tables(
    symbol_tables: Sequence[tuple[KNode[T], ...]],
) -> tuple[tuple[KNode[T], ...], list[list[int]]]:
    """
    Merges multiple symbol tables into a unified table.

    Returns (unified_table, mappings) where mappings[i][j] gives the new index
    for symbol j from table i. Selects optimal symbols per equivalence class
    (preferring abstractions, then smallest bit length).
    """
    equivalence_classes: defaultdict[KNode, list[tuple[KNode, int]]] = defaultdict(list)
    dependency_graph: defaultdict[tuple[KNode, int], set[int]] = defaultdict(set)

    # Collect resolved symbols and build dependency graph
    for table_idx, table in enumerate(symbol_tables):
        for symbol in table:
            resolved_symbol = resolve_symbols(symbol, table)
            equivalence_classes[resolved_symbol].append((symbol, table_idx))
            subsymbols = set(index.value for index in contained_symbols(symbol))
            dependency_graph[(symbol, table_idx)] |= subsymbols

    # Select optimal symbol per equivalence class
    selected_symbols: dict[KNode, KNode] = {}
    for resolved, symbols_in_class in equivalence_classes.items():
        abstracted = [s for s, _ in symbols_in_class if is_abstraction(s)]
        if abstracted:
            selected = max(abstracted, key=lambda s: (arity(s), -s.bit_length()))
        else:
            selected = min(
                (s for s, _ in symbols_in_class), key=lambda s: s.bit_length()
            )
        selected_symbols[resolved] = selected

    # Build reverse dependency graph for topological sort
    reverse_deps: defaultdict[KNode, set[KNode]] = defaultdict(set)
    symbol_to_selected: dict[tuple[KNode, int], KNode] = {}

    for resolved, selected in selected_symbols.items():
        for symbol, table_idx in equivalence_classes[resolved]:
            symbol_to_selected[(symbol, table_idx)] = selected
            table = symbol_tables[table_idx]
            for dep_idx in dependency_graph[(symbol, table_idx)]:
                if dep_idx < len(table):
                    dep_symbol = table[dep_idx]
                    dep_resolved = resolve_symbols(dep_symbol, table)
                    dep_selected = selected_symbols[dep_resolved]
                    reverse_deps[dep_selected].add(selected)

    for s in selected_symbols.values():
        if s not in reverse_deps:
            reverse_deps[s] = set()

    unified_symbols = list(topological_sort(reverse_deps))
    symbol_to_index = {s: i for i, s in enumerate(unified_symbols)}

    mappings: list[list[int]] = [[] for _ in symbol_tables]
    for table_idx, table in enumerate(symbol_tables):
        for symbol in table:
            resolved = resolve_symbols(symbol, table)
            selected = selected_symbols[resolved]
            mappings[table_idx].append(symbol_to_index[selected])

    return tuple(unified_symbols), mappings


def symbolize_together(
    trees: tuple[KNode[T], ...], symbol_tables: Sequence[tuple[KNode[T], ...]]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Merges symbol tables and re-symbolizes trees with the unified table.

    Steps:
        1. Merge symbol tables with index remapping
        2. Update tree SymbolNode indices
        3. Factor trees against unified symbols
        4. Find new common patterns across all trees
    """
    if symbol_tables:
        if len(symbol_tables) != len(trees):
            raise ValueError(
                f"There are only {len(symbol_tables)} symbol tables for {len(trees)} trees"
            )

        unified_symbols, mappings = merge_symbol_tables(symbol_tables)

        updated_trees = tuple(
            remap_symbol_indices(tree, mapping, i)
            for i, (tree, mapping) in enumerate(zip(trees, mappings))
        )

        factored_trees = tuple(
            factor_by_existing_symbols(tree, unified_symbols) for tree in updated_trees
        )
    else:
        factored_trees = trees
        unified_symbols = tuple()

    final_trees, final_symbols = symbolize(factored_trees, unified_symbols)
    return final_trees, final_symbols


def unsymbolize(knode: KNode[T], symbol_table: Sequence[KNode[T]]) -> KNode[T]:
    """
    Expands all SymbolNodes and NestedNodes using the symbol table.

    Requires the symbol table to contain all referenced templates
    and to be fully resolved (no inter-symbol references).
    """
    nnode = copy.deepcopy(knode)
    nnode = resolve_symbols(nnode, symbol_table)
    nnode = expand_all_nested_nodes(nnode, symbol_table)
    return nnode


def unsymbolize_all(
    trees: Sequence[KNode[T]], symbol_table: Sequence[KNode[T]]
) -> tuple[KNode[T], ...]:
    """
    Unsymbolizes multiple trees.

    Assumes no cycles in symbol table (symbols only reference later symbols).
    """
    symbol_table = tuple(symbol_table)
    resolved_table = tuple(resolve_symbols(symb, symbol_table) for symb in symbol_table)
    return tuple(unsymbolize(tree, resolved_table) for tree in trees)


def full_symbolization(
    trees: Sequence[KNode[T]],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Complete symbolization pipeline: nesting extraction then pattern symbolization.
    """
    symbol_table: list[KNode[T]] = []
    nested = tuple(
        extract_nested_patterns(symbol_table, syntax_tree) for syntax_tree in trees
    )

    symbolized, symbol_table_out = symbolize(tuple(nested), tuple(symbol_table))
    return symbolized, symbol_table_out
