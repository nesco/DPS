"""
Symbolization utilities for Kolmogorov Tree.

This module provides functions for finding common patterns in trees,
symbolizing them, merging symbol tables, and unsymbolizing trees.
"""

import copy
from collections import Counter, defaultdict, deque
from typing import Sequence

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
    Identifies frequent concrete and abstracted subtrees across multiple KolmogorovTrees.
    Returns the top patterns ranked by bit-length savings.
    """
    # Step 1: Collect all subtrees
    all_subtrees: list[KNode[T]] = []
    for tree in trees:
        all_subtrees.extend(breadth_first_preorder_knode(tree))

    # Step 2: Count frequencies, track matches, and cache template extractions
    pattern_counter: Counter[KNode[T]] = Counter()
    pattern_matches: defaultdict[KNode[T], list[KNode[T]]] = defaultdict(list)
    # Cache: subtree -> list of (abstract_pattern, params) tuples
    template_cache: dict[KNode[T], list[tuple[KNode[T], tuple]]] = {}

    for subtree in all_subtrees:
        pattern_counter[subtree] += 1
        pattern_matches[subtree].append(subtree)

        # Cache template extraction results
        if subtree not in template_cache:
            template_cache[subtree] = list(extract_template(subtree))

        for abs_pattern, params in template_cache[subtree]:
            pattern_counter[abs_pattern] += 1
            pattern_matches[abs_pattern].append(subtree)

    # Step 3: Filter patterns
    common_patterns: list[KNode[T]] = []
    seen_patterns: set[KNode[T]] = set()

    for pattern, count in pattern_counter.items():
        if count < min_occurrences or pattern in seen_patterns:
            continue
        if any(
            isinstance(n, VariableNode) for n in breadth_first_preorder_knode(pattern)
        ):
            if len(pattern_matches[pattern]) >= min_occurrences:
                common_patterns.append(pattern)
                seen_patterns.add(pattern)
        else:
            common_patterns.append(pattern)
            seen_patterns.add(pattern)

    # Step 4: Calculate bit gain and filter for positive savings
    # Pre-compute param_len for each pattern using the cache
    pattern_param_len: dict[KNode[T], float] = {}
    for pat in common_patterns:
        count = pattern_counter[pat]
        total_param_bits = 0
        for s in pattern_matches[pat]:
            # Use cached template extraction
            for cached_pat, ps in template_cache.get(s, []):
                if pat == cached_pat:
                    total_param_bits += sum(p.bit_length() for p in ps)
        pattern_param_len[pat] = total_param_bits / count if count > 0 else 0

    def bit_gain(pat: KNode[T]) -> float:
        count = pattern_counter[pat]
        current_len = sum(s.bit_length() for s in pattern_matches[pat])
        param_len = pattern_param_len.get(pat, 0)
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        return current_len - (count * symb_len + pat.bit_length())

    # TODO: Currently repeats of simple moves by the same count of a move are not symbolized,
    # because the SymbolNode is heavier
    # so a lot of repeat by the same count won't be symbolized

    # Might need
    # and pattern_counter[node] > tree_count (lattice_count)
    # You may want more than 1 per full tree ?
    common_patterns = [pat for pat in common_patterns if bit_gain(pat) > 0]

    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]


def symbolize_pattern(
    trees: Sequence[KNode[T]],
    symbols: Sequence[KNode[T]],
    new_symbol: KNode[T],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
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
    Symbolize trees by replacing the common pattern with SymbolNodes.
    It only do the most common pattern, because symbolizing one pattern potentially changes the bit length saving of the rest
    """

    # While there is a pattern whose abstraction leads to bit gain savings
    common_patterns = find_symbol_candidates(trees + symbols)
    while common_patterns:
        # Abstract the best one
        new_symbol = common_patterns[0]
        trees, symbols = symbolize_pattern(trees, symbols, new_symbol)
        common_patterns = find_symbol_candidates(trees + symbols)

    return (trees, symbols)


def symbolize(
    trees: tuple[KNode[T], ...], symbols: tuple[KNode[T], ...]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    i = 0
    # Phase 1: Non-symbolic patterns
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not is_symbolized(c) and not isinstance(c, RootNode)
        ]
        # print(f"candidates len: {len(candidates)}")
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])
        i += 1

    # # Phase 2: Symbolic patterns by depth
    # for depth in range(1, max_depth(trees) + 1):
    #     while True:
    #         candidates = [c for c in find_symbol_candidates(trees + symbols) if len(contained_symbols(c)) <= depth]
    #         if not candidates:
    #             break
    #         trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    # Phase 2: Including Symbolic patterns:
    while True:
        candidates = [
            c
            for c in find_symbol_candidates(trees + symbols)
            if not isinstance(c, RootNode)
        ]
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])
        i += 1

    # Phase 3: Include roots
    while True:
        candidates = find_symbol_candidates(
            trees + symbols
        )  # All candidates, including RootNodes
        if not candidates:
            break
        trees, symbols = symbolize_pattern(trees, symbols, candidates[0])

    return trees, symbols


## Re-factorization:


def factor_by_existing_symbols(
    tree: KNode[T], symbols: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Factors a tree against an existing symbol table, replacing matches with SymbolNodes.

    Args:
        tree: The tree to factor.
        symbols: The symbol table to check against.

    Returns:
        Factored tree with applicable SymbolNode replacements.
    """

    def factor_node(node: KNode[T]) -> KNode[T]:
        for i, symbol in enumerate(symbols):
            abstracted = abstract_node(IndexValue(i), symbol, node)
            if abstracted != node:  # If abstraction occurred
                return abstracted
        return node

    return postmap(tree, factor_node, factorize=True)


def remap_symbol_indices(tree: KNode[T], mapping: list[int], tree_idx: int) -> KNode[T]:
    """
    Updates SymbolNode indices in a tree based on the provided mapping.
    Used to remap the tree elements.

    Args:
        tree: The tree to update.
        mapping: List mapping old indices to new ones.
        tree_idx: Index of the tree for debugging purposes.

    Returns:
        Updated tree with remapped SymbolNode indices.
    """

    def update_node(node: KNode[T]) -> KNode[T]:
        if isinstance(node, SymbolNode) and node.index.value < len(mapping):
            new_index = IndexValue(mapping[node.index.value])
            return SymbolNode(new_index, node.parameters)
        return node

    return postmap(tree, update_node, factorize=False)


def remap_sub_symbols(
    symbol: KNode[T], mapping: list[int], original_table: tuple[KNode[T], ...]
) -> KNode[T]:
    """
    Remaps SymbolNode indices within a symbol based on the provided mapping.
    Used to remap the symbol tables elements.

    Args:
        symbol: The symbol to update.
        mapping: List mapping old indices to new ones.
        original_table: The original symbol table for resolution if needed.

    Returns:
        Updated symbol with remapped SymbolNode indices.
    """

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
    Merges multiple symbol tables into a unified table, returning the table and mappings.
    It choose symbols so to minimize the total bit length.
    Note: Clean resymbolisation is often preferable.

    Args:
        symbol_tables: List of symbol tables to merge.

    Returns:
        Tuple of the unified symbol table and a list of mappings (old index -> new index) for each input table.
    """
    unified_symbols: list[KNode] = []
    mappings: list[list[int]] = [[] for _ in range(len(symbol_tables))]

    equivalence_classes: defaultdict[KNode, list[tuple[KNode, int]]] = defaultdict(
        list
    )  # Store symbol classes, in an union-find like structure. Each classes contains the symbols and their tables
    dependency_graph: defaultdict[tuple[KNode, int], set[int]] = defaultdict(
        set
    )  # Store for each symbol the symbols it depends on

    # Step 1: Collect all resolved symbolsa and build the dependency graph
    for i, table in enumerate(symbol_tables):
        for symbol in table:
            resolved_symbol = resolve_symbols(symbol, table)
            equivalence_classes[resolved_symbol].append((symbol, i))
            # Track dependencies: which symbols this symbol references in its table
            subsymbols = set(index.value for index in contained_symbols(symbol))
            dependency_graph[(symbol, i)] |= subsymbols

    # Step 2: Select optimal symbols per equivalence class
    selected_symbols = {}
    for resolved, symbols_in_class in equivalence_classes.items():
        # Prefer abstracted symbols (those with variables)
        abstracted = [s for s, _ in symbols_in_class if is_abstraction(s)]
        if abstracted:
            # Select the one with the most variables, then smallest bit_length
            selected = max(abstracted, key=lambda s: (arity(s), -s.bit_length()))
        else:
            # Select the one with the smallest bit_length
            selected = min(
                (s for s, _ in symbols_in_class), key=lambda s: s.bit_length()
            )
        selected_symbols[resolved] = selected

    # Step 3: Update dependency graph with selected symbols
    new_dependency_graph = defaultdict(set)
    symbol_to_selected = {}
    for resolved, selected in selected_symbols.items():
        for symbol, table_idx in equivalence_classes[resolved]:
            symbol_to_selected[(symbol, table_idx)] = selected
            table = symbol_tables[table_idx]
            for dep_idx in dependency_graph[(symbol, table_idx)]:
                if dep_idx < len(table):
                    dep_symbol = table[dep_idx]
                    dep_resolved = resolve_symbols(dep_symbol, table)
                    dep_selected = selected_symbols[dep_resolved]
                    new_dependency_graph[selected].add(dep_selected)

    # Step 4: Topological sort to order symbols in the unified table
    in_degree = {s: len(deps) for s, deps in new_dependency_graph.items()}
    for s in selected_symbols.values():
        if s not in in_degree:
            in_degree[s] = 0
    queue = deque([s for s, deg in in_degree.items() if deg == 0])
    unified_symbols = []

    while queue:
        symbol = queue.popleft()
        unified_symbols.append(symbol)
        for dependent in new_dependency_graph:
            if symbol in new_dependency_graph[dependent]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    symbol_to_index = {s: i for i, s in enumerate(unified_symbols)}
    mappings = [[] for _ in symbol_tables]

    for table_idx, table in enumerate(symbol_tables):
        for symbol in table:
            resolved = resolve_symbols(symbol, table)
            selected = selected_symbols[resolved]
            mappings[table_idx].append(symbol_to_index[selected])
    # TODO

    return tuple(unified_symbols), mappings


def symbolize_together(
    trees: tuple[KNode[T], ...], symbol_tables: Sequence[tuple[KNode[T], ...]]
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Merges independently symbolized trees into a unified symbol table and re-symbolizes them together.

    Args:
        trees: Tuple of KNode[MoveValue] trees, each potentially containing SymbolNodes.
        symbol_tables: List of symbol tables corresponding to each tree (or empty if unsymbolized).

    Returns:
        Tuple of updated trees and the unified symbol table.
    """
    if symbol_tables:
        if len(symbol_tables) != len(trees):
            raise ValueError(
                f"There are only {len(symbol_tables)} symbol tables for {len(trees)} trees"
            )
        # Step 1: Merge symbol tables into a unified table with index remapping
        unified_symbols, mappings = merge_symbol_tables(symbol_tables)

        # Step 2: Update trees with the new symbol indices
        updated_trees = tuple(
            remap_symbol_indices(tree, mapping, i)
            for i, (tree, mapping) in enumerate(zip(trees, mappings))
        )

        # Step 3: Factor existing patterns against the unified symbol table
        factored_trees = tuple(
            factor_by_existing_symbols(tree, unified_symbols) for tree in updated_trees
        )
    else:
        factored_trees = trees
        unified_symbols = tuple()

    # Step 4: Find and symbolize new common patterns across all trees
    final_trees, final_symbols = symbolize(factored_trees, unified_symbols)

    return final_trees, final_symbols


def unsymbolize(knode: KNode[T], symbol_table: Sequence[KNode[T]]) -> KNode[T]:
    """
    Completely unsymbolize a given node.
    If the symbol table contains all the referenced templates, the resulting node
    should be free of any SymbolNodes or NestedNodes
    Second hypothesis: the symbol table has been completely resolved too.

    Args:
        knode: KNode containing NestedNodes and SymbolNodes.
        symbol_table: Symbol table containing all the templates referenced by NestedNodes and SymbolNodes
    """
    nnode = copy.deepcopy(knode)
    # Step 1: First unsymbolize SymbolNode
    nnode = resolve_symbols(nnode, symbol_table)

    # Step 2: Then unsymbolize NestedNodes
    nnode = expand_all_nested_nodes(nnode, symbol_table)

    return nnode


def unsymbolize_all(
    trees: Sequence[KNode[T]], symbol_table: Sequence[KNode[T]]
) -> tuple[KNode[T], ...]:
    """
    Hypothesis: no loop in the symbol table, a symbol can only reference a symbol strictly after him
    """
    # Step 1: Resolve the symbol table
    symbol_table = tuple(symbol_table)
    nsymbol_table = []
    for i, symb in enumerate(symbol_table):
        nsymbol_table.append(resolve_symbols(symb, symbol_table))

    nsymbol_table = tuple(nsymbol_table)

    # Step 2: Unsymbolize all the nodes
    return tuple(unsymbolize(tree, nsymbol_table) for tree in trees)


def full_symbolization(
    trees: Sequence[KNode[T]],
) -> tuple[tuple[KNode[T], ...], tuple[KNode[T], ...]]:
    """
    Full standard symbolization + node nesting.
    """
    # Step 1: nest nodes
    symbol_table: list[KNode[T]] = []
    nested = tuple(
        extract_nested_patterns(symbol_table, syntax_tree) for syntax_tree in trees
    )

    symbolized, symbol_table_out = symbolize(tuple(nested), tuple(symbol_table))

    return symbolized, symbol_table_out
