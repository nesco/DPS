"""
Symbolization for Kolmogorov Tree.

Functions:
    Discovery:
        find_symbol_candidates(trees)   - Find frequent patterns ranked by bit savings

    Symbolization:
        symbolize_pattern(trees, symbols, pattern) - Replace pattern with SymbolNode
        greedy_symbolization(trees, symbols)       - Iteratively symbolize best patterns
        symbolize(trees, symbols)                  - Full multi-phase symbolization
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
from typing import Sequence

from utils.dag_functionals import topological_sort

from kolmogorov_tree.matching import abstract_node, node_to_symbolized_node
from kolmogorov_tree.nodes import (
    KNode,
    RootNode,
    SymbolNode,
    VariableNode,
)
from kolmogorov_tree.predicates import (
    arity,
    contained_symbols,
    is_abstraction,
    is_symbolized,
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


def find_symbol_candidates(
    trees: Sequence[KNode[T]],
    min_occurrences: int = 2,
    max_patterns: int = 10,
) -> list[KNode]:
    """
    Finds frequent subtrees ranked by bit-length savings.

    Returns top patterns where symbolization reduces total bit length.
    Considers both concrete matches and abstracted patterns (with variables).
    """
    all_subtrees: list[KNode[T]] = []
    for tree in trees:
        all_subtrees.extend(breadth_first_preorder_knode(tree))

    pattern_counter: Counter[KNode[T]] = Counter()
    pattern_matches: defaultdict[KNode[T], list[KNode[T]]] = defaultdict(list)
    template_cache: dict[KNode[T], list[tuple[KNode[T], tuple]]] = {}

    for subtree in all_subtrees:
        pattern_counter[subtree] += 1
        pattern_matches[subtree].append(subtree)

        if subtree not in template_cache:
            template_cache[subtree] = list(extract_template(subtree))

        for abs_pattern, params in template_cache[subtree]:
            pattern_counter[abs_pattern] += 1
            pattern_matches[abs_pattern].append(subtree)

    common_patterns: list[KNode[T]] = []
    seen_patterns: set[KNode[T]] = set()

    for pattern, count in pattern_counter.items():
        if count < min_occurrences or pattern in seen_patterns:
            continue

        has_variables = any(
            isinstance(n, VariableNode) for n in breadth_first_preorder_knode(pattern)
        )

        if has_variables:
            if len(pattern_matches[pattern]) >= min_occurrences:
                common_patterns.append(pattern)
                seen_patterns.add(pattern)
        else:
            common_patterns.append(pattern)
            seen_patterns.add(pattern)

    # Pre-compute average parameter bit length per pattern
    pattern_param_len: dict[KNode[T], float] = {}
    for pat in common_patterns:
        count = pattern_counter[pat]
        total_param_bits = 0
        for s in pattern_matches[pat]:
            for cached_pat, ps in template_cache.get(s, []):
                if pat == cached_pat:
                    total_param_bits += sum(p.bit_length() for p in ps)
        pattern_param_len[pat] = total_param_bits / count if count > 0 else 0

    def bit_gain(pat: KNode[T]) -> float:
        count = pattern_counter[pat]
        current_len = sum(s.bit_length() for s in pattern_matches[pat])
        param_len = pattern_param_len.get(pat, 0)
        symbol_overhead = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        return current_len - (count * symbol_overhead + pat.bit_length())

    # Filter to patterns with positive bit savings
    common_patterns = [pat for pat in common_patterns if bit_gain(pat) > 0]
    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]


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


def greedy_symbolization(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Repeatedly symbolizes the highest-gain pattern until no savings remain.

    Only symbolizes one pattern per iteration since each symbolization
    changes bit-length calculations for remaining candidates.
    """
    common_patterns = find_symbol_candidates(trees + symbols)
    while common_patterns:
        new_symbol = common_patterns[0]
        trees, symbols = symbolize_pattern(trees, symbols, new_symbol)
        common_patterns = find_symbol_candidates(trees + symbols)

    return (trees, symbols)


def symbolize(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Multi-phase symbolization with increasing pattern complexity.

    Phase 1: Non-symbolic patterns (primitives, products, sums)
    Phase 2: Patterns containing SymbolNodes
    Phase 3: RootNode patterns
    """
    # Phase 1: Non-symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not is_symbolized(c) and not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 2: Symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 3: RootNode patterns
    while True:
        candidates = find_symbol_candidates(trees + symbols)
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

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
    resolved_table = tuple(
        resolve_symbols(symb, symbol_table) for symb in symbol_table
    )
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
