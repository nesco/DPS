"""
Tests for the kolmogorov_tree module.

These tests were extracted from the original kolmogorov_tree.py monolithic file.
"""

from kolmogorov_tree import (
    # Primitives
    BitLength,
    ARCBitLength,
    CountValue,
    VariableValue,
    IndexValue,
    NoneValue,
    MoveValue,
    PaletteValue,
    CoordValue,
    # Nodes
    KNode,
    PrimitiveNode,
    VariableNode,
    ProductNode,
    SumNode,
    RepeatNode,
    NestedNode,
    SymbolNode,
    RootNode,
    # Functions
    shift,
    encode_run_length,
    construct_product_node,
    get_iterator,
    find_repeating_pattern,
    factorize_tuple,
    is_abstraction,
    arity,
    extract_nested_sum_template,
    extract_nested_product_template,
    nested_collection_to_nested_node,
    extract_template,
    symbolize_pattern,
    resolve_symbols,
    find_symbol_candidates,
    matches,
    unify,
    Bindings,
    abstract_node,
    node_to_symbolized_node,
    expand_nested_node,
    factor_by_existing_symbols,
    substitute_variables,
    remap_symbol_indices,
    remap_sub_symbols,
    merge_symbol_tables,
    symbolize_together,
    create_move_node,
    create_variable_node,
)
from localtypes import Coord, ensure_all_instances


# Helpers for tests
def root_to_symbolize():
    # Step 1: Define Primitive Nodes
    move0 = PrimitiveNode(MoveValue(0))
    move1 = PrimitiveNode(MoveValue(1))
    move2 = PrimitiveNode(MoveValue(2))
    move3 = PrimitiveNode(MoveValue(3))
    move5 = PrimitiveNode(MoveValue(5))
    move6 = PrimitiveNode(MoveValue(6))
    move7 = PrimitiveNode(MoveValue(7))

    # Step 2: Define Repeat Nodes
    repeat0 = RepeatNode(move0, CountValue(4))
    repeat1 = RepeatNode(move1, CountValue(4))
    repeat2 = RepeatNode(move2, CountValue(4))
    repeat3 = RepeatNode(move3, CountValue(4))

    # Step 3: Build the Nested Structure
    level3 = ProductNode((move7, repeat3))  # "7(3)*{4}"
    level2 = SumNode(frozenset({repeat2, level3}))  # "[(2)*{4}|7(3)*{4}]"
    level1 = ProductNode((move6, level2))  # "6[(2)*{4}|7(3)*{4}]"
    level0 = SumNode(
        frozenset({repeat1, level1})
    )  # "[(1)*{4}|6[(2)*{4}|7(3)*{4}]]"
    level_1 = ProductNode((move5, level0))  # "5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]"
    level_2 = SumNode(
        frozenset({repeat0, level_1})
    )  # "[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]]"
    program = ProductNode(
        (move0, level_2)
    )  # "0[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]]"

    # Step 4: Create the Root Node
    root_node = RootNode(
        program, CoordValue(Coord(5, 5)), PaletteValue(frozenset({1}))
    )

    # Verify the string representation
    # print(
    #     str(root_node)
    # )  # Should output: "Root(0[(0)*{4}|5[(1)*{4}|6[(2)*{4}|7(3)*{4}]]], (5, 5), {1})"

    return root_node


# Tests
def test_encode_run_length():
    # Test Case 1: Empty Input
    result = encode_run_length([])
    assert result == ProductNode(()), (
        "Test Case 1 Failed: Empty input should return empty ProductNode"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = encode_run_length([node])
    assert result == node, (
        "Test Case 2 Failed: Single node should not be wrapped in ProductNode"
    )

    # Test Case 3: No Repeats
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 3 Failed: No repeats should remain uncompressed"
    )

    # Test Case 4: Short Repeats
    nodes = [
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
    ]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 4 Failed: Repeats less than 3 should not be compressed"
    )

    # Test Case 5: Long Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = encode_run_length(nodes)
    expected = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3))
    assert result == expected, (
        "Test Case 5 Failed: Three identical nodes should be compressed"
    )

    # Test Case 6: Mixed Sequences
    nodes = (
        [PrimitiveNode(MoveValue(1))] * 3
        + [PrimitiveNode(MoveValue(2))]
        + [PrimitiveNode(MoveValue(3))] * 2
    )
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    assert result == expected, (
        "Test Case 6 Failed: Mixed sequence compression incorrect"
    )

    # Test Case 7: Multiple Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [
        PrimitiveNode(MoveValue(2))
    ] * 4
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 7 Failed: Multiple repeat sequences not handled correctly"
    )

    # Test Case 8: All Identical
    nodes = [PrimitiveNode(MoveValue(5))] * 10
    result = encode_run_length(nodes)
    expected = RepeatNode(PrimitiveNode(MoveValue(5)), CountValue(10))
    assert result == expected, (
        "Test Case 8 Failed: All identical nodes should compress into one RepeatNode"
    )

    # Test Case 9: Input as Iterator
    nodes = iter([PrimitiveNode(MoveValue(1))] * 4)
    result = encode_run_length(nodes)
    expected = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(4))
    assert result == expected, (
        "Test Case 9 Failed: Iterator input not handled correctly"
    )

    # Test Case 10: Alternating Nodes
    nodes = [
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(1)),
        PrimitiveNode(MoveValue(2)),
    ]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), (
        "Test Case 11 Failed: Alternating nodes should not be compressed"
    )

    # Test Case 11: Repeats in Different Positions
    nodes = (
        [PrimitiveNode(MoveValue(1))] * 3
        + [PrimitiveNode(MoveValue(2))]
        + [PrimitiveNode(MoveValue(3))] * 3
        + [PrimitiveNode(MoveValue(4))] * 2
    )
    result = encode_run_length(nodes)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
            RepeatNode(PrimitiveNode(MoveValue(3)), CountValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 12 Failed: Repeats in different positions not handled correctly"
    )

    print("Test 5: `encode_run_length` tests - Passed")


def test_construct_product_node():
    # Test 1: Empty Input
    # Verifies that an empty iterable returns an empty ProductNode
    result = construct_product_node([])
    expected = ProductNode(())
    assert result == expected, (
        "Test 1 Failed: Empty input should return empty ProductNode"
    )

    # Test 2: Single Node
    # Checks that a single node is wrapped in a ProductNode without changes
    node = PrimitiveNode(MoveValue(1))
    result = construct_product_node([node])
    expected = ProductNode((node,))
    assert result == expected, (
        "Test 2 Failed: Single node should be wrapped in ProductNode"
    )

    # Test 3: Merging Adjacent PrimitiveNodes
    # Ensures consecutive identical PrimitiveNodes are compressed into a RepeatNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = construct_product_node(nodes)
    expected = ProductNode(
        (RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),)
    )
    assert result == expected, (
        "Test 3 Failed: Adjacent PrimitiveNodes should merge into RepeatNode"
    )

    # Test 4: Preserving SumNodes
    # Confirms that SumNodes are kept intact and not merged
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(3)), PrimitiveNode(MoveValue(4))])
    )
    nodes = [PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2))]
    result = construct_product_node(nodes)
    expected = ProductNode(
        (PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2)))
    )
    assert result == expected, "Test 4 Failed: SumNodes should be preserved"
    print("Test 6 - Product Node consstructor basic tests passed successfully!")


def test_shift():
    """
    Tests the shift operation across different node types in the KolmogorovTree.
    The shift operation should modify MoveValue primitives by adding k modulo 8,
    while leaving non-MoveValue primitives unchanged.
    """
    # Test 1: Shifting a PrimitiveNode with MoveValue
    node = create_move_node(2)
    shifted = shift(node, 1)
    assert isinstance(shifted, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted.data == 3, "MoveValue should shift from 2 to 3 with k=1"

    # Test 2: Shifting by 0 (no change)
    shifted_zero = shift(node, 0)
    assert shifted_zero == node, "Shifting by 0 should return the same node"
    assert isinstance(shifted_zero, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_zero.data == 2, "Value should remain 2 when k=0"

    # Test 3: Shifting SumNode
    sum_node = SumNode(frozenset([create_move_node(0), create_move_node(4)]))
    shifted_sum = shift(sum_node, 2)
    assert isinstance(shifted_sum, SumNode), (
        "Shifted result should be a SumNode"
    )
    assert len(shifted_sum.children) == 2, "SumNode should retain 2 children"
    primitive_children = ensure_all_instances(
        shifted_sum.children, PrimitiveNode
    )

    data_values = {child.data for child in primitive_children}
    assert data_values == {2, 6}, "Children should have data values 2 and 6"

    # Test 4: Shifting RepeatNode
    sequence = ProductNode((create_move_node(2), create_move_node(3)))
    repeat = RepeatNode(sequence, CountValue(3))
    shifted_repeat = shift(repeat, 1)
    assert isinstance(shifted_repeat, RepeatNode), (
        "Shifted result should be a RepeatNode"
    )
    assert isinstance(shifted_repeat.node, ProductNode), (
        "Repeated node should be a ProductNode"
    )
    assert isinstance(shifted_repeat.node.children[0], PrimitiveNode), (
        "First child should be PrimitiveNode"
    )
    assert shifted_repeat.node.children[0].data == 3, (
        "First MoveValue should shift to 3"
    )
    assert isinstance(shifted_repeat.node.children[1], PrimitiveNode), (
        "Second child should be PrimitiveNode"
    )
    assert shifted_repeat.node.children[1].data == 4, (
        "Second MoveValue should shift to 4"
    )
    assert isinstance(shifted_repeat.count, CountValue), (
        "COunt should be CountValue"
    )
    assert shifted_repeat.count.value == 3, "Count should remain unchanged"

    # Test 5: Shifting SymbolNode
    param1 = create_move_node(1)
    param2 = PaletteValue(frozenset({2}))
    symbol = SymbolNode(IndexValue(0), (param1, param2))
    shifted_symbol = shift(symbol, 1)
    assert isinstance(shifted_symbol, SymbolNode), (
        "Shifted result should be a SymbolNode"
    )
    assert len(shifted_symbol.parameters) == 2, (
        "SymbolNode should retain 2 parameters"
    )
    assert isinstance(shifted_symbol.parameters[0], PrimitiveNode), (
        "Parameter should be PrimitiveNode"
    )
    assert shifted_symbol.parameters[0].data == 2, (
        "MoveValue parameter should shift to 2"
    )
    assert shifted_symbol.parameters[1] is param2, (
        "Non-shiftable parameter should be unchanged"
    )

    # Test 6: Shifting RootNode
    program = ProductNode((create_move_node(0), create_move_node(1)))
    root = RootNode(
        program, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    shifted_root = shift(root, 2)
    assert isinstance(shifted_root, RootNode), (
        "Shifted result should be a RootNode"
    )
    assert isinstance(shifted_root.node, ProductNode), (
        "Root program should be a ProductNode"
    )
    assert isinstance(shifted_root.node.children[0], PrimitiveNode), (
        "First child should be PrimitiveNode"
    )
    assert shifted_root.node.children[0].data == 2, (
        "First MoveValue should shift to 2"
    )
    assert isinstance(shifted_root.node.children[1], PrimitiveNode), (
        "Second child should be PrimitiveNode"
    )
    assert shifted_root.node.children[1].data == 3, (
        "Second MoveValue should shift to 3"
    )
    assert shifted_root.position == CoordValue(Coord(0, 0)), (
        "Position should be unchanged"
    )
    assert shifted_root.colors == PaletteValue(frozenset({1})), (
        "Colors should be unchanged"
    )

    # Test 7: Shifting with large k (wrapping around)
    node_large = create_move_node(7)
    shifted_large = shift(node_large, 10)  # 7 + 10 = 17 ≡ 1 mod 8
    assert isinstance(shifted_large, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_large.data == 1, "Large shift should wrap around to 1"

    # Test 8: Shifting with negative k
    shifted_neg = shift(node_large, -3)  # 7 - 3 = 4
    assert isinstance(shifted_neg, PrimitiveNode), (
        "Shifted node should be a PrimitiveNode"
    )
    assert shifted_neg.data == 4, "Negative shift should result in 4"

    # Test 9: Shifting nested composite nodes
    inner_product = ProductNode((create_move_node(5), create_move_node(6)))
    repeat_inner = RepeatNode(inner_product, CountValue(2))
    primitive_outer = create_move_node(7)
    outer_sum = SumNode(frozenset([repeat_inner, primitive_outer]))
    shifted_outer = shift(outer_sum, 1)

    # Verify the shifted outer node is a SumNode
    assert isinstance(shifted_outer, SumNode), (
        "Shifted outer node should be a SumNode"
    )

    # Extract children from frozenset based on type
    children_list = list(shifted_outer.children)
    repeat_nodes = [
        child for child in children_list if isinstance(child, RepeatNode)
    ]
    primitive_nodes = [
        child for child in children_list if isinstance(child, PrimitiveNode)
    ]

    # Ensure the correct number of each type
    assert len(repeat_nodes) == 1, (
        "There should be one RepeatNode in the SumNode"
    )
    assert len(primitive_nodes) == 1, (
        "There should be one PrimitiveNode in the SumNode"
    )

    # Get the single RepeatNode and PrimitiveNode
    shifted_repeat = repeat_nodes[0]
    shifted_primitive = primitive_nodes[0]

    # Verify RepeatNode properties
    assert isinstance(shifted_repeat, RepeatNode), (
        "Child should be a RepeatNode"
    )
    assert isinstance(shifted_repeat.node, ProductNode), (
        "Repeated node should be ProductNode"
    )
    assert len(shifted_repeat.node.children) == 2, (
        "ProductNode should have two children"
    )

    # Add assertion to narrow child types to PrimitiveNode

    primitive_children = ensure_all_instances(
        shifted_repeat.node.children, PrimitiveNode
    )
    data_values = [child.data for child in primitive_children]
    assert data_values == [6, 7], "Nested MoveValues should shift to 6 and 7"

    # Verify PrimitiveNode properties
    assert isinstance(shifted_primitive, PrimitiveNode), (
        "Child should be a PrimitiveNode"
    )
    assert shifted_primitive.data == 0, (
        "Outer MoveValue should shift from 7 to 0"
    )
    # Test 10: Original node remains unchanged
    original_node = create_move_node(2)
    shifted_node = shift(original_node, 1)
    assert isinstance(original_node, PrimitiveNode), (
        "Original node should be PrimitiveNode"
    )
    assert original_node.data == 2, "Original node value should remain 2"
    assert isinstance(shifted_node, PrimitiveNode), (
        "Shifted node should be PrimitiveNode"
    )
    assert shifted_node.data == 3, "Shifted node value should be 3"

    print("Test shift operations - Passed")


def test_get_iterator():
    """Tests the get_iterator function for detecting and compressing arithmetic sequences."""
    # Test Case 1: Empty Input
    result = get_iterator([])
    assert result == frozenset(), (
        "Test Case 1 Failed: Empty input should return an empty frozenset"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = get_iterator([node])
    assert result == frozenset([node]), (
        "Test Case 2 Failed: Single node should return a frozenset with that node"
    )

    # Test Case 3: Sequence with Increment +1
    nodes_pos = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]]
    result_pos = get_iterator(nodes_pos)
    expected_pos_forward = frozenset((RepeatNode(nodes_pos[0], CountValue(3)),))
    expected_pos_backward = frozenset(
        (RepeatNode(nodes_pos[2], CountValue(-3)),)
    )
    assert result_pos in [expected_pos_forward, expected_pos_backward], (
        "Test Case 3 Failed: Sequence [0,1,2] should be compressed to RepeatNode(0, 3) or RepeatNode(2, -3)"
    )

    # Test Case 4: Sequence with Increment -1
    # Note: Since frozenset loses order, both [2,1,0] and [0,1,2] produce the same set {0,1,2}
    # Either compression is valid: RepeatNode(0, 3) or RepeatNode(2, -3)
    nodes_neg = [PrimitiveNode(MoveValue(i)) for i in [2, 1, 0]]
    result_neg = get_iterator(nodes_neg)
    expected_neg_forward = frozenset((RepeatNode(nodes_neg[2], CountValue(3)),))  # Start from 0
    expected_neg_backward = frozenset((RepeatNode(nodes_neg[0], CountValue(-3)),))  # Start from 2
    assert result_neg in [expected_neg_forward, expected_neg_backward], (
        "Test Case 4 Failed: Sequence [2,1,0] should be compressed to RepeatNode(0, 3) or RepeatNode(2, -3)"
    )

    # Test Case 5: Non-Sequence
    nodes_non = [PrimitiveNode(MoveValue(i)) for i in [0, 5, 1]]
    result_non = get_iterator(nodes_non)
    assert result_non == frozenset(nodes_non), (
        "Test Case 5 Failed: Non-sequence should return original nodes"
    )

    # Test Case 6: Boundary Conditions (Wrap-around with Increment +1)
    nodes_wrap = [PrimitiveNode(MoveValue(i)) for i in [7, 0, 1]]
    result_wrap = get_iterator(nodes_wrap)
    expected_wrap_forward = frozenset(
        (RepeatNode(nodes_wrap[0], CountValue(3)),)
    )
    expected_wrap_backward = frozenset(
        (RepeatNode(nodes_wrap[2], CountValue(-3)),)
    )
    assert result_wrap in [expected_wrap_forward, expected_wrap_backward], (
        "Test Case 6 Failed: Wrap-around sequence [7,0,1] should be compressed"
    )

    # Test Case 7: Long Sequence with Wrap-around, should be equal to the size of the alphabet
    nodes_long = [
        PrimitiveNode(MoveValue(i % 8)) for i in range(10)
    ]  # [0,1,2,3,4,5,6,7,0,1]
    result_long = get_iterator(nodes_long)

    # Add assertion to narrow the type to RepeatNode
    assert len(result_long) == 1 and isinstance(
        next(iter(result_long)), RepeatNode
    ), "Expected a single children"

    node = next(iter(result_long))
    assert isinstance(node, RepeatNode) and node.count == CountValue(8), (
        "Test Case 7 Failed: Count should be 8"
    )

    # Test Case 8: Partial Sequence
    nodes_partial = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2, 4]]
    result_partial = get_iterator(nodes_partial)
    assert result_partial == frozenset(nodes_partial), (
        "Test Case 8 Failed: Partial sequence [0,1,2,4] should not be compressed"
    )

    # Test Case 9: Different Increment
    nodes_diff_inc = [PrimitiveNode(MoveValue(i)) for i in [0, 2, 4, 6]]
    result_diff_inc = get_iterator(nodes_diff_inc)
    assert result_diff_inc == frozenset(nodes_diff_inc), (
        "Test Case 9 Failed: Sequence with increment +2 should not be compressed"
    )

    # Test Case 10: Sequence of RepeatNodes with same count
    r0 = RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(5))
    r1 = RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(5))
    r2 = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(5))
    r3 = RepeatNode(PrimitiveNode(MoveValue(3)), CountValue(5))
    nodes = [r0, r1, r2, r3]
    result = get_iterator(nodes)
    assert len(result) == 1, (
        "Test Case 10 Failed: Should return a frozenset with one RepeatNode"
    )
    repeat_node = next(iter(result))
    assert isinstance(repeat_node, RepeatNode), (
        "Test Case 10 Failed: Result should be a RepeatNode"
    )
    assert repeat_node.node in nodes, (
        "Test Case 10 Failed: The node should be one of the original nodes"
    )
    assert isinstance(repeat_node.count, CountValue) and abs(
        repeat_node.count.value
    ) == len(nodes), (
        "Test Case 10 Failed: The count should equal the number of nodes"
    )
    # Verify that shifts regenerate the original set
    if repeat_node.count.value > 0:
        shifts = range(repeat_node.count.value)
    else:
        shifts = range(0, repeat_node.count.value, -1)
    expected = {shift(repeat_node.node, k) for k in shifts}
    assert expected == frozenset(nodes), (
        "Test Case 10 Failed: Shifts should regenerate the original set"
    )

    print("Test get_iterator - Passed")


def test_find_repeating_pattern():
    """Tests the find_repeating_pattern function for detecting repeating patterns."""
    # Test Case 1: Empty Input
    result = find_repeating_pattern([], 0)
    assert result == (None, 0, False), (
        "Test Case 1 Failed: Empty input should return (None, 0, False)"
    )

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = find_repeating_pattern([node], 0)
    assert result == (None, 0, False), (
        "Test Case 2 Failed: Single node should return (None, 0, False)"
    )

    # Test Case 3: Simple Repeat
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert pattern == nodes[0], (
        "Test Case 3 Failed: Pattern should be MoveValue(2)"
    )
    assert count == 3, "Test Case 3 Failed: Count should be 3"
    assert not is_reversed, "Test Case 3 Failed: Should not be reversed"

    # Test Case 4: Alternating Repeat
    nodes = [
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
    ]
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert isinstance(pattern, ProductNode) and str(pattern) == "[2,0]", (
        "Test Case 4 Failed: Pattern should be 20"
    )
    assert count == -3, "Test Case 4 Failed: Count should be -3 for alternating"
    assert is_reversed, "Test Case 4 Failed: Should be reversed"

    # Test Case 5: Multi-Node Pattern
    pattern_nodes = [PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3))]
    nodes = pattern_nodes * 3  # 2,3,2,3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert isinstance(pattern, ProductNode) and pattern.children == tuple(
        pattern_nodes
    ), "Test Case 5 Failed: Incorrect pattern"
    assert count == 3, "Test Case 5 Failed: Count should be 3"
    assert not is_reversed, "Test Case 5 Failed: Should not be reversed"

    # Test Case 6: No Repeat
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = find_repeating_pattern(nodes, 0)
    assert result == (None, 0, False), (
        "Test Case 6 Failed: Non-repeating sequence should return (None, 0, False)"
    )

    # Test Case 7: Offset Pattern
    nodes = [PrimitiveNode(MoveValue(1))] + [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 1)
    assert pattern == PrimitiveNode(MoveValue(2)), (
        "Test Case 7 Failed: Pattern should be MoveValue(2)"
    )
    assert count == 3, "Test Case 7 Failed: Count should be 3"
    assert not is_reversed, "Test Case 7 Failed: Should not be reversed"

    print("Test find_repeating_pattern - Passed")


def test_factorize_tuple():
    """Tests the factorize_tuple function for compressing ProductNode and SumNode."""
    # Test Case 1: Empty ProductNode
    node = ProductNode(())
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 1 Failed: Empty ProductNode should remain unchanged"
    )

    # Test Case 2: Single Node ProductNode
    node = ProductNode((PrimitiveNode(MoveValue(1)),))
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 2 Failed: Single node ProductNode should remain unchanged"
    )

    # Test Case 3: Repeating ProductNode
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3))
    assert result == expected, (
        f"Test Case 3 Failed: Repeating nodes should be compressed. Got {[result]} instead of {[expected]}"
    )

    # Test Case 4: Alternating ProductNode
    nodes = [
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(0)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(0)),
    ]

    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(0)))),
        CountValue(-3),
    )
    assert result == expected, (
        "Test Case 4 Failed: Alternating pattern should be compressed with negative count"
    )

    # Test Case 5: Mixed ProductNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))]
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
            PrimitiveNode(MoveValue(2)),
        )
    )
    assert result == expected, (
        "Test Case 5 Failed: Mixed sequence compression incorrect"
    )

    # Test Case 6: SumNode Arithmetic Sequence
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]])
    )
    result = factorize_tuple(sum_node)
    expected_forward = SumNode(
        frozenset((RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(3)),))
    )
    expected_backward = SumNode(
        frozenset((RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(-3)),))
    )
    assert result in [expected_forward, expected_backward], (
        "Test Case 6 Failed: SumNode arithmetic sequence should be compressed"
    )

    # Test Case 7: SumNode Non-Sequence
    sum_node = SumNode(
        frozenset([PrimitiveNode(MoveValue(i)) for i in [0, 5, 1]])
    )
    result = factorize_tuple(sum_node)
    assert result == sum_node, (
        "Test Case 7 Failed: Non-sequence SumNode should remain unchanged"
    )

    # Test Case 8: Non-Product/Sum Node
    node = PrimitiveNode(MoveValue(1))
    result = factorize_tuple(node)
    assert result == node, (
        "Test Case 8 Failed: Non-Product/Sum node should remain unchanged"
    )

    print("Test factorize_tuple - Passed")


def test_is_abstraction():
    """Tests the is_abstraction function for detecting VariableNodes in the tree."""
    # Test Case 1: Node is a VariableNode
    var_node = VariableNode(VariableValue(0))
    assert is_abstraction(var_node), (
        "Test Case 1 Failed: VariableNode itself should return True"
    )

    # Test Case 2: Node is a PrimitiveNode (no VariableNode)
    prim_node = PrimitiveNode(MoveValue(1))
    assert not is_abstraction(prim_node), (
        "Test Case 2 Failed: PrimitiveNode should return False"
    )

    # Test Case 3: ProductNode with one VariableNode child
    var_child = VariableNode(VariableValue(1))
    tuple_node = ProductNode((prim_node, var_child))
    assert is_abstraction(tuple_node), (
        "Test Case 3 Failed: ProductNode with VariableNode child should return True"
    )

    # Test Case 4: ProductNode with no VariableNodes
    tuple_no_var = ProductNode((prim_node, PrimitiveNode(MoveValue(2))))
    assert not is_abstraction(tuple_no_var), (
        "Test Case 4 Failed: ProductNode without VariableNodes should return False"
    )

    # Test Case 5: RepeatNode with VariableNode in subtree
    repeat_node = RepeatNode(var_child, CountValue(2))
    assert is_abstraction(repeat_node), (
        "Test Case 5 Failed: RepeatNode with VariableNode should return True"
    )

    # Test Case 6: SymbolNode with VariableNode as parameter
    symbol_with_var = SymbolNode(IndexValue(0), (var_child,))
    assert is_abstraction(symbol_with_var), (
        "Test Case 6 Failed: SymbolNode with VariableNode parameter should return True"
    )

    # Test Case 7: RootNode with VariableNode in program
    root_with_var = RootNode(
        var_child, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    assert is_abstraction(root_with_var), (
        "Test Case 7 Failed: RootNode with VariableNode in program should return True"
    )

    # Test Case 8: Node with only non-KNode BitLengthAware subvalues
    # PrimitiveNode has a Primitive subvalue (e.g., MoveValue), which is BitLengthAware but not a VariableNode
    assert not is_abstraction(prim_node), (
        "Test Case 8 Failed: Node with only non-KNode subvalues should return False"
    )

    print("Test is_abstraction - Passed")


def test_arity():
    """
    Test function for the `arity` function, which computes the number of parameters
    in a Kolmogorov Tree pattern based on the highest variable index plus one.
    """
    # Test Case 1: Single variable with index 0
    node1 = VariableNode(VariableValue(0))
    assert arity(node1) == 1, (
        "Test Case 1 Failed: Expected arity 1 for VariableNode(0)"
    )

    # Test Case 2: No variables
    node2 = PrimitiveNode(MoveValue(2))
    assert arity(node2) == 0, (
        "Test Case 2 Failed: Expected arity 0 for no variables"
    )

    # Test Case 3: Two variables with indices 0 and 1
    node3 = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(1)))
    )
    assert arity(node3) == 2, (
        f"Test Case 3 Failed: Expected arity 2 for indices [0, 1], got {arity(node3)}"
    )

    # Test Case 4: Single variable in a RepeatNode
    node4 = RepeatNode(VariableNode(VariableValue(0)), CountValue(4))
    assert arity(node4) == 1, (
        "Test Case 4 Failed: Expected arity 1 for index [0]"
    )

    # Test Case 5: Variables with indices 0 and 2 in a SumNode
    node5 = SumNode(
        frozenset(
            {VariableNode(VariableValue(0)), VariableNode(VariableValue(2))}
        )
    )
    assert arity(node5) == 3, (
        "Test Case 5 Failed: Expected arity 3 for indices [0, 2]"
    )

    # Test Case 6: Variable in a NestedNode
    node6 = NestedNode(
        IndexValue(0), VariableNode(VariableValue(1)), CountValue(3)
    )
    assert arity(node6) == 2, (
        "Test Case 6 Failed: Expected arity 2 for index [1]"
    )

    # Test Case 7: Variable in a SymbolNode's parameters
    node7 = SymbolNode(IndexValue(0), (VariableNode(VariableValue(0)),))
    assert arity(node7) == 1, (
        "Test Case 7 Failed: Expected arity 1 for index [0]"
    )

    # Test Case 8: No variables in a RepeatNode
    node8 = RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(4))
    assert arity(node8) == 0, (
        "Test Case 8 Failed: Expected arity 0 for no variables"
    )

    # Test Case 9: Variables in nested structure
    node9 = ProductNode(
        (
            VariableNode(VariableValue(0)),
            RepeatNode(VariableNode(VariableValue(1)), CountValue(3)),
        )
    )
    assert arity(node9) == 2, (
        "Test Case 9 Failed: Expected arity 2 for indices [0, 1]"
    )

    # Test Case 10: Same variable used multiple times
    node10 = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(0)))
    )
    assert arity(node10) == 1, (
        "Test Case 10 Failed: Expected arity 1 for index [0]"
    )

    print("All arity tests passed successfully!")


def test_extract_nested_sum_template():
    """Tests the extract_nested_sum_template function."""

    def create_move_node(value):
        return PrimitiveNode(MoveValue(value))

    a, b, c, d, e, f, g, h = [create_move_node(i) for i in range(8)]
    inner_sum1 = SumNode(frozenset([b, c]))
    inner_sum2 = SumNode(frozenset([f, g]))
    prod1 = ProductNode((a, inner_sum1, d))
    prod2 = ProductNode((e, inner_sum2, h))
    main_sum = SumNode(frozenset([prod1, prod2]))
    result = extract_nested_sum_template(main_sum)
    assert result is not None, "Expected a template and parameter"
    template, parameter = result
    var_node = VariableNode(VariableValue(0))
    expected_template1 = SumNode(
        frozenset([ProductNode((a, var_node, d)), prod2])
    )
    expected_template2 = SumNode(
        frozenset([prod1, ProductNode((e, var_node, h))])
    )
    if template.children == expected_template1.children:
        assert parameter == inner_sum1, "Parameter should be inner_sum1"
    elif template.children == expected_template2.children:
        assert parameter == inner_sum2, "Parameter should be inner_sum2"
    else:
        assert False, f"Unexpected template structure: {str(template)}"
    sum_no_product = SumNode(frozenset([a, b, c]))
    assert extract_nested_sum_template(sum_no_product) is None, "Expected None"
    prod_no_sum = ProductNode((a, b, c))
    sum_with_prod_no_sum = SumNode(frozenset([prod_no_sum]))
    assert extract_nested_sum_template(sum_with_prod_no_sum) is None, (
        "Expected None"
    )
    print("Test extract_nested_sum_template - Passed")


def test_extract_nested_product_template():
    """Tests the extract_nested_product_template function."""

    def create_move_node(value):
        return PrimitiveNode(MoveValue(value))

    a, b, c, d, e, f = [create_move_node(i) for i in range(6)]
    inner_prod1 = ProductNode((b, c))
    inner_prod2 = ProductNode((d, e))
    sum_inner = SumNode(frozenset([inner_prod1, inner_prod2]))
    main_product = ProductNode((a, sum_inner, f))
    result = extract_nested_product_template(main_product)
    assert result is not None, "Expected a template and parameter"
    template, parameter = result
    var_node = VariableNode(VariableValue(0))
    expected_template1 = ProductNode(
        (a, SumNode(frozenset([var_node, inner_prod2])), f)
    )
    expected_template2 = ProductNode(
        (a, SumNode(frozenset([inner_prod1, var_node])), f)
    )
    if template.children == expected_template1.children:
        assert parameter == inner_prod1, "Parameter should be inner_prod1"
    elif template.children == expected_template2.children:
        assert parameter == inner_prod2, "Parameter should be inner_prod2"
    else:
        assert False, f"Unexpected template structure: {str(template)}"
    product_no_sum = ProductNode((a, b, c))
    assert extract_nested_product_template(product_no_sum) is None, (
        "Expected None"
    )
    sum_no_product = SumNode(frozenset([a, b]))
    product_with_sum = ProductNode((sum_no_product, c))
    assert extract_nested_product_template(product_with_sum) is None, (
        "Expected None"
    )
    print("Test extract_nested_product_template - Passed")


def test_node_to_symbolize():
    node = SumNode(
        frozenset(
            {
                SymbolNode(IndexValue(4), ()),
                RootNode(
                    SymbolNode(IndexValue(0), ()),
                    CoordValue(Coord(4, 4)),
                    PaletteValue(frozenset({3})),
                ),
                SymbolNode(IndexValue(6), ()),
                SymbolNode(
                    IndexValue(3),
                    (CoordValue(Coord(6, 0)),),
                ),
                SymbolNode(IndexValue(5), ()),
            }
        )
    )
    pattern = SumNode(
        frozenset(
            {
                SymbolNode(IndexValue(4), ()),
                SymbolNode(IndexValue(6), ()),
                VariableNode(VariableValue(0)),
                SymbolNode(
                    IndexValue(3),
                    (CoordValue(Coord(6, 0)),),
                ),
                SymbolNode(
                    IndexValue(5),
                    (
                        # RootNode(
                        #     SymbolNode(IndexValue(0), ()),
                        #     CoordValue(Coord(4, 4)),
                        #     PaletteValue(frozenset({3})),
                        # ),
                    ),
                ),
            }
        )
    )

    symbolized = node_to_symbolized_node(IndexValue(6), pattern, node)
    s2 = abstract_node(IndexValue(6), pattern, node)
    print(f"s2: {s2}")

    expected = SymbolNode(
        IndexValue(6),
        (
            RootNode(
                SymbolNode(IndexValue(0), ()),
                CoordValue(Coord(4, 4)),
                PaletteValue(frozenset({3})),
            ),
        ),
    )

    assert symbolized == expected, f"{symbolized} != {expected}"


def test_nested_collection_to_nested_node():
    """
    Tests the `nested_collection_to_nested_node` function for capturing recursive patterns in SumNodes.
    Verifies template extraction, terminal node identification, recursion count, and reconstruction.
    """

    # Helper functions to expand NestedNode for testing
    def substitute(template, substitution):
        """Substitutes VariableNodes in a template with given nodes."""
        if isinstance(template, SumNode):
            new_children = frozenset(
                substitute(child, substitution) for child in template.children
            )
            return SumNode(new_children)
        elif isinstance(template, ProductNode):
            new_children = tuple(
                substitute(child, substitution) for child in template.children
            )
            return ProductNode(new_children)
        elif isinstance(template, VariableNode):
            return substitution[template.index.value]
        return template

    def expand_nested(template, terminal, count):
        """Expands a NestedNode by applying the template `count` times starting from the terminal."""
        current = terminal
        for _ in range(count):
            current = substitute(template, {0: current})
        return current

    # Test Case 1: Recursive SumNode with multiple levels
    # Pattern: [0 Rec | 4], repeated 3 times for simplicity (scalable to 8 as in your example)
    terminal = SumNode(
        frozenset([PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))])
    )
    level1 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), terminal)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )
    level2 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), level1)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )
    level3 = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), level2)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )

    result = nested_collection_to_nested_node(level3)
    assert result is not None, (
        "Expected a NestedNode and template for recursive SumNode"
    )
    nested_node, template = result

    # Expected template: [0 Var(0) | 4]
    var_node = VariableNode(VariableValue(0))
    expected_template = SumNode(
        frozenset(
            [
                ProductNode((PrimitiveNode(MoveValue(0)), var_node)),
                PrimitiveNode(MoveValue(4)),
            ]
        )
    )

    # Verify template
    assert template == expected_template, (
        f"Template mismatch: expected {expected_template}, got {template}"
    )

    # Verify NestedNode properties
    assert isinstance(nested_node, NestedNode), "Result should be a NestedNode"
    assert nested_node.node == terminal, (
        f"Terminal node mismatch: expected {terminal}, got {nested_node.node}"
    )
    assert isinstance(nested_node.count, CountValue)

    assert nested_node.count.value == 3, (
        f"Count mismatch: expected 3, got {nested_node.count.value}"
    )
    # Index is a placeholder (0), not critical for this test

    # Verify reconstruction
    expanded = expand_nested(
        template, nested_node.node, nested_node.count.value
    )
    assert expanded == level3, "Expanded node should match original level3"

    # Test Case 2: Non-recursive SumNode
    non_recursive = SumNode(
        frozenset([PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))])
    )
    result = nested_collection_to_nested_node(non_recursive)
    assert result is None, "Expected None for non-recursive SumNode"

    # Test Case 3: Recursive SumNode with single level
    result = nested_collection_to_nested_node(level1)
    assert result is not None, (
        "Expected a NestedNode for single-level recursive SumNode"
    )
    nested_node, template = result
    assert template == expected_template, (
        "Template mismatch in single-level case"
    )
    assert isinstance(nested_node.count, CountValue)
    assert nested_node.count.value == 1, (
        f"Count should be 1, got {nested_node.count.value}"
    )
    assert nested_node.node == terminal, (
        "Terminal node mismatch in single-level case"
    )
    expanded = expand_nested(
        template, nested_node.node, nested_node.count.value
    )
    assert expanded == level1, "Expanded node should match original level1"

    print("Test nested_collection_to_nested_node - Passed")


def test_symbolize_pattern():
    root_node = root_to_symbolize()

    # Create PrimitiveNodes
    move0 = PrimitiveNode(MoveValue(0))
    move5 = PrimitiveNode(MoveValue(5))
    move6 = PrimitiveNode(MoveValue(6))
    move7 = PrimitiveNode(MoveValue(7))

    # Create SymbolNodes
    s0_3 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(3)),))
    s0_2 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(2)),))
    s0_1 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(1)),))
    s0_0 = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(0)),))

    # Build the tree bottom-up
    inner_sum = SumNode(frozenset({ProductNode((move7, s0_3)), s0_2}))
    prod6 = ProductNode((move6, inner_sum))
    sum2 = SumNode(frozenset({prod6, s0_1}))
    prod5 = ProductNode((move5, sum2))
    sum3 = SumNode(frozenset({prod5, s0_0}))
    program = ProductNode((move0, sum3))

    # Create the RootNode
    root_symbolized = RootNode(
        program, CoordValue(Coord(5, 5)), PaletteValue(frozenset({1}))
    )

    symbol = RepeatNode(VariableNode(VariableValue(0)), CountValue(4))
    r_symb, sym_table = symbolize_pattern((root_node,), tuple(), symbol)
    assert len(r_symb) == 1
    assert len(sym_table) == 1
    assert r_symb[0] == root_symbolized
    assert sym_table[0] == symbol


def test_resolve_symbols():
    # Test Case 1: Test that a tree with no symbols remains unchanged.
    node = PrimitiveNode(MoveValue(1))
    symbols: tuple[KNode, ...] = ()
    result = resolve_symbols(node, symbols)
    assert result == node, "Tree with no symbols should remain unchanged"

    # Test Case 2: Test resolving a single SymbolNode without parameters.
    symbol_def = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    symbols = (symbol_def,)
    node = SymbolNode(IndexValue(0), ())
    result = resolve_symbols(node, symbols)
    assert result == symbol_def, (
        "SymbolNode should be replaced by its definition"
    )

    # Test Case 3: Test resolving a single SymbolNode with parameters.
    symbol_def = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(1)))
    )
    symbols = (symbol_def,)
    param = PrimitiveNode(MoveValue(4))
    node = SymbolNode(IndexValue(0), (param,))
    expected = ProductNode((param, PrimitiveNode(MoveValue(1))))
    result = resolve_symbols(node, symbols)
    assert result == expected, "Parameters should be substituted correctly"

    # Test Case 4: Test resolving a composite node containing a SymbolNode.
    symbol_def = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(1)))
    )
    symbols = (symbol_def,)
    param = PrimitiveNode(MoveValue(5))
    symbol_node = SymbolNode(IndexValue(0), (param,))
    composite = ProductNode((symbol_node, PrimitiveNode(MoveValue(6))))
    expected_inner = ProductNode((param, PrimitiveNode(MoveValue(1))))
    expected = ProductNode((expected_inner, PrimitiveNode(MoveValue(6))))
    result = resolve_symbols(composite, symbols)
    assert result == expected, (
        "Composite node should resolve its children correctly"
    )

    # Test Case 5: Test resolving nested symbols
    symbol0 = PrimitiveNode(MoveValue(7))
    symbol1 = ProductNode(
        (SymbolNode(IndexValue(0), ()), PrimitiveNode(MoveValue(8)))
    )
    symbols = (
        symbol0,
        symbol1,
    )
    node = SymbolNode(IndexValue(1), ())
    expected = ProductNode((symbol0, PrimitiveNode(MoveValue(8))))
    result = resolve_symbols(node, symbols)
    assert result == expected, "Nested symbols should be resolved recursively"

    # Test Case 6: Test resolving a symbol with parameters in a RepeatNode.
    symbol_def = RepeatNode(
        PrimitiveNode(MoveValue(2)), VariableNode(VariableValue(0))
    )
    symbols = (symbol_def,)
    param = CountValue(3)
    node = SymbolNode(IndexValue(0), (param,))
    expected = RepeatNode(PrimitiveNode(MoveValue(2)), param)
    result = resolve_symbols(node, symbols)
    assert result == expected, (
        f"Parameters should substitute into RepeatNode's count: {expected}, {result}"
    )

    # Test Case 7: Test that a SymbolNode with an invalid index remains unchanged.
    symbols = (PrimitiveNode(MoveValue(1)),)
    node = SymbolNode(IndexValue(1), ())
    result = resolve_symbols(node, symbols)
    assert result == node, (
        "SymbolNode with invalid index should remain unchanged"
    )

    # Test Case 8: Test resolving a tree with multiple symbols and parameters.
    symbol0 = PrimitiveNode(MoveValue(3))
    symbol1 = ProductNode(
        (SymbolNode(IndexValue(0), ()), VariableNode(VariableValue(0)))
    )
    symbols = (symbol0, symbol1)
    param = PrimitiveNode(MoveValue(4))
    node = SymbolNode(IndexValue(1), (param,))
    expected = ProductNode((symbol0, param))
    result = resolve_symbols(node, symbols)
    assert result == expected, (
        "Multiple symbols and parameters should be handled correctly"
    )
    print("Test resolve_symbols - Passed")


def test_find_symbol_candidates():
    """
    Tests the find_symbol_candidates function for identifying frequent subtrees
    across multiple Kolmogorov Trees, ensuring correct handling of frequency,
    bit-length savings, abstraction, edge cases, and integration.
    """

    # Test Case 1: Identical Repeating Pattern with Positive Bit Gain
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(2)))
    )  # bit_length = 15
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    tree3 = RootNode(
        pattern, CoordValue(Coord(2, 2)), PaletteValue(frozenset({3}))
    )
    tree4 = RootNode(
        pattern, CoordValue(Coord(3, 3)), PaletteValue(frozenset({4}))
    )
    trees = (tree1, tree2, tree3, tree4)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 2, (
        "Test Case 1 Failed: Should find exactly two candidate"
    )
    assert candidates[0] == RootNode(
        pattern, VariableNode(VariableValue(0)), VariableNode(VariableValue(1))
    ), "Test Case 1 Failed: Candidate should be the repeating pattern"
    print("Test Case 1: Basic Functionality - Passed")

    # Test Case 2: Frequency Threshold - Pattern Below Threshold
    unique_tree = RootNode(
        PrimitiveNode(MoveValue(3)),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    trees = (tree1, unique_tree)  # Pattern appears only once
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 0, (
        "Test Case 2 Failed: Should find no candidates below frequency threshold"
    )
    print("Test Case 2: Frequency Threshold - Passed")

    # Test Case 3: Bit-Length Savings - Exclude Non-Saving Patterns
    short_pattern = PrimitiveNode(MoveValue(2))  # Bit length too small to save
    tree5 = RootNode(
        short_pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree6 = RootNode(
        short_pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    trees = (tree5, tree6)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    # Short pattern (bit_length=6) won't save bits vs. SymbolNode (bit_length=10)
    assert len(candidates) == 0, (
        "Test Case 3 Failed: Should exclude patterns with no bit savings"
    )
    print("Test Case 3: Bit-Length Savings - Passed")

    # Test Case 4: Abstraction Handling - Abstracted Pattern
    tree7 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(1)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    tree8 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(3)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue(Coord(1, 1)),
        PaletteValue(frozenset({2})),
    )
    tree9 = RootNode(
        ProductNode(
            (
                PrimitiveNode(MoveValue(5)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
                PrimitiveNode(MoveValue(2)),
            )
        ),
        CoordValue(Coord(2, 2)),
        PaletteValue(frozenset({3})),
    )
    trees = (tree7, tree8, tree9)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    expected_abs = ProductNode(
        (
            VariableNode(VariableValue(0)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(2)),
        )
    )
    assert any(candidate == expected_abs for candidate in candidates), (
        "Test Case 4 Failed: Should include abstracted pattern"
    )
    print("Test Case 4: Abstraction Handling - Passed")

    # Test Case 5a: Edge Case - Empty Input
    candidates = find_symbol_candidates((), min_occurrences=2, max_patterns=5)
    assert len(candidates) == 0, (
        "Test Case 5a Failed: Empty input should return empty list"
    )
    print("Test Case 5a: Edge Case (Empty Input) - Passed")

    # Test Case 5b: Edge Case - No Common Patterns
    tree10 = RootNode(
        PrimitiveNode(MoveValue(1)),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    tree11 = RootNode(
        PrimitiveNode(MoveValue(2)),
        CoordValue(Coord(1, 1)),
        PaletteValue(frozenset({2})),
    )
    trees = (tree10, tree11)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=5
    )
    assert len(candidates) == 0, (
        "Test Case 5b Failed: No common patterns should return empty list"
    )
    print("Test Case 5b: Edge Case (No Common Patterns) - Passed")

    # Test Case 6: Interaction with Other Functions - Symbolization and Resolution
    trees = (tree1, tree2, tree3, tree4)
    candidates = find_symbol_candidates(
        trees, min_occurrences=2, max_patterns=1
    )
    assert len(candidates) == 1, (
        "Test Case 6 Failed: Should find one candidate for symbolization"
    )
    symbol_index = IndexValue(0)
    symbolized_trees = tuple(
        abstract_node(symbol_index, candidates[0], tree) for tree in trees
    )
    symbols = (candidates[0],)
    resolved_trees = tuple(
        resolve_symbols(tree, symbols) for tree in symbolized_trees
    )
    assert resolved_trees == trees, (
        "Test Case 6 Failed: Resolved trees should match original trees"
    )
    print("Test Case 6: Interaction with Other Functions - Passed")

    print("All test_find_symbol_candidates tests - Passed")


def test_matching():
    # Test Case 1: No Variables
    node = PrimitiveNode(MoveValue(2))
    pattern = node
    bindings = matches(pattern, node)
    assert bindings == {}, (
        "Expected empty bindings for exact match without variables"
    )

    # Test Case 2: With Variables
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(3)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    expected_bindings = {0: PrimitiveNode(MoveValue(2))}
    assert bindings == expected_bindings, (
        f"Expected bindings {expected_bindings}, got {bindings}"
    )

    # Test Case 3: No match
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(4)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    assert bindings is None, "Expected no match, but got bindings"

    # Test Case 4: Abstraction and Resolution
    original_node = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), PrimitiveNode(MoveValue(3)))
    )
    parameters = (PrimitiveNode(MoveValue(2)),)
    index = IndexValue(0)

    abstracted_node = abstract_node(index, pattern, original_node)
    assert isinstance(abstracted_node, SymbolNode), (
        "Expected SymbolNode after abstraction"
    )
    assert abstracted_node.index == index, "Index mismatch"
    assert abstracted_node.parameters == parameters, (
        f"Expected parameters {parameters}"
    )

    symbols = (pattern,)
    resolved_node = resolve_symbols(abstracted_node, symbols)
    assert resolved_node == original_node, (
        "Resolved node does not match original"
    )

    # Test Case 5: Nested Structures
    nested_node = ProductNode(
        (
            RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    pattern = ProductNode(
        (
            RepeatNode(VariableNode(VariableValue(0)), CountValue(3)),
            PrimitiveNode(MoveValue(3)),
        )
    )
    parameters = (PrimitiveNode(MoveValue(2)),)
    index = IndexValue(0)

    abstracted_node = abstract_node(index, pattern, nested_node)
    assert isinstance(abstracted_node, SymbolNode), "Expected SymbolNode"
    assert abstracted_node.parameters == parameters, "Parameters mismatch"

    symbols = (pattern,)
    resolved_node = resolve_symbols(abstracted_node, symbols)
    assert resolved_node == nested_node, (
        "Resolved nested node does not match original"
    )

    # Tes Case 6: Multiple Variables
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(1)))
    )
    subtree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings = matches(pattern, subtree)
    expected_bindings = {
        0: PrimitiveNode(MoveValue(2)),
        1: PrimitiveNode(MoveValue(3)),
    }
    assert bindings == expected_bindings, (
        f"Expected bindings {expected_bindings}, got {bindings}"
    )

    # Tes Case 7: Variable Reuse
    pattern = ProductNode(
        (VariableNode(VariableValue(0)), VariableNode(VariableValue(0)))
    )
    subtree1 = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(2)))
    )
    bindings1 = matches(pattern, subtree1)
    assert bindings1 == {0: PrimitiveNode(MoveValue(2))}, (
        "Expected binding for identical elements"
    )

    subtree2 = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    bindings2 = matches(pattern, subtree2)
    assert bindings2 is None, "Expected no match for different elements"

    # Test Case 8: Integration with extract_template
    node = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    abstractions = extract_template(node)

    for pattern, params in abstractions:
        index = IndexValue(0)
        abstracted_node = abstract_node(index, pattern, node)
        if abstracted_node != node:  # Abstraction occurred
            symbols = (pattern,)
            resolved_node = resolve_symbols(abstracted_node, symbols)
            assert resolved_node == node, (
                f"Resolved node does not match original for pattern {pattern}"
            )

    print("Test matching - Passed")


def test_unify_sum():
    """
    Tests the unify function specifically for the SumNode case,
    focusing on correctness and deterministic binding.
    """

    # Test Case 1: Exact Match (No Variables), different frozenset order
    pattern1 = SumNode(
        frozenset(
            {create_move_node(1), create_move_node(2), create_move_node(3)}
        )
    )
    subtree1a = SumNode(
        frozenset(
            {create_move_node(3), create_move_node(1), create_move_node(2)}
        )
    )
    subtree1b = SumNode(
        frozenset(
            {create_move_node(1), create_move_node(2), create_move_node(3)}
        )
    )
    bindings1a: Bindings = {}
    bindings1b: Bindings = {}
    assert unify(pattern1, subtree1a, bindings1a)
    assert bindings1a == {}, "Exact match should produce empty bindings"
    assert unify(pattern1, subtree1b, bindings1b)
    assert bindings1b == {}, (
        "Exact match should produce empty bindings regardless of order"
    )

    # Test Case 2: Mismatch - Different Number of Children
    pattern2 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    subtree2 = SumNode(
        frozenset(
            {create_move_node(1), create_move_node(2), create_move_node(3)}
        )
    )
    bindings2: Bindings = {}
    assert not unify(pattern2, subtree2, bindings2), (
        "Should fail on different child count"
    )
    assert bindings2 == {}, "Bindings should be empty after failed unification"

    # Test Case 3: Mismatch - Same Number, Different Content
    pattern3 = SumNode(
        frozenset({create_move_node(1), create_move_node(4)})
    )  # create_move_node(4) instead of create_move_node(2)
    subtree3 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    bindings3: Bindings = {}
    assert not unify(pattern3, subtree3, bindings3), (
        "Should fail on different content"
    )
    assert bindings3 == {}, "Bindings should be empty after failed unification"

    # Test Case 4: Single Variable - Successful Binding
    pattern4 = SumNode(
        frozenset({create_variable_node(0), create_move_node(2)})
    )  # Var(0) and create_move_node(2)
    subtree4 = SumNode(
        frozenset({create_move_node(1), create_move_node(2)})
    )  # Match create_move_node(2), Var(0) should bind to create_move_node(1)
    bindings4: Bindings = {}
    assert unify(pattern4, subtree4, bindings4), (
        "Single variable unification should succeed"
    )
    # Deterministic check: create_move_node(2) matches create_move_node(2). create_variable_node(0) must match the remaining create_move_node(1).
    assert bindings4 == {0: create_move_node(1)}, (
        f"Expected Var(0) to bind to create_move_node(1), got {bindings4}"
    )

    # Test Case 5: Single Variable - Binding to a Complex Node
    prod_node = ProductNode((create_move_node(3), create_move_node(4)))
    pattern5 = SumNode(
        frozenset({create_move_node(1), create_variable_node(0)})
    )
    subtree5 = SumNode(frozenset({create_move_node(1), prod_node}))
    bindings5: Bindings = {}
    assert unify(pattern5, subtree5, bindings5), (
        "Variable binding to ProductNode should succeed"
    )
    # create_move_node(1) matches create_move_node(1). create_variable_node(0) must bind to the remaining prod_node.
    assert bindings5 == {0: prod_node}, (
        f"Expected Var(0) to bind to {prod_node}, got {bindings5}"
    )

    # Test Case 6: Multiple Variables - Successful Binding & Determinism
    pattern6 = SumNode(
        frozenset(
            {
                create_move_node(5),
                create_variable_node(1),
                create_variable_node(0),
            }
        )
    )
    subtree6a = SumNode(
        frozenset(
            {create_move_node(3), create_move_node(5), create_move_node(4)}
        )
    )  # Deliberate order difference
    subtree6b = SumNode(
        frozenset(
            {create_move_node(4), create_move_node(3), create_move_node(5)}
        )
    )  # Another order
    bindings6a: Bindings = {}
    bindings6b: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted: ['5', 'Var(0)', 'Var(1)']
    # Subtree sorted: ['3', '4', '5']
    # Match '5' -> '5'.
    # Match 'Var(0)' -> '3'.
    # Match 'Var(1)' -> '4'.
    expected_bindings6 = {0: create_move_node(3), 1: create_move_node(4)}

    assert unify(pattern6, subtree6a, bindings6a), (
        "Multi-variable unification (a) should succeed"
    )
    assert bindings6a == expected_bindings6, (
        f"Bindings (a) mismatch: Expected {expected_bindings6}, got {bindings6a}"
    )

    assert unify(pattern6, subtree6b, bindings6b), (
        "Multi-variable unification (b) should succeed"
    )
    assert bindings6b == expected_bindings6, (
        f"Bindings (b) mismatch: Expected {expected_bindings6}, got {bindings6b}"
    )
    assert bindings6a == bindings6b, (
        "Bindings should be deterministic regardless of initial frozenset order"
    )

    # Test Case 7: Multiple Variables - Failure due to Content Mismatch
    pattern7 = SumNode(
        frozenset({create_variable_node(0), create_move_node(2)})
    )
    subtree7 = SumNode(
        frozenset({create_move_node(1), create_move_node(3)})
    )  # No create_move_node(2) to match the concrete part
    bindings7: Bindings = {}
    assert not unify(pattern7, subtree7, bindings7), (
        "Should fail if concrete parts don't match"
    )
    assert bindings7 == {}, "Bindings should be empty after failed unification"

    # Test Case 8: Determinism Check with Swapped Variables in Pattern
    pattern8a = SumNode(
        frozenset({create_variable_node(0), create_variable_node(1)})
    )  # Var(0), Var(1)
    pattern8b = SumNode(
        frozenset({create_variable_node(1), create_variable_node(0)})
    )  # Var(1), Var(0) - same set
    subtree8 = SumNode(frozenset({create_move_node(1), create_move_node(2)}))
    bindings8a: Bindings = {}
    bindings8b: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted (both cases): ['Var(0)', 'Var(1)']
    # Subtree sorted: ['1', '2']
    # Match 'Var(0)' -> '1'.
    # Match 'Var(1)' -> '2'.
    expected_bindings8 = {0: create_move_node(1), 1: create_move_node(2)}

    assert unify(pattern8a, subtree8, bindings8a)
    assert bindings8a == expected_bindings8, (
        f"Bindings (8a) mismatch: Expected {expected_bindings8}, got {bindings8a}"
    )
    # Even though pattern8b looks different, its frozenset is identical
    assert unify(pattern8b, subtree8, bindings8b)
    assert bindings8b == expected_bindings8, (
        f"Bindings (8b) mismatch: Expected {expected_bindings8}, got {bindings8b}"
    )

    # Test Case 9: Nested SumNode Unification
    inner_p9 = SumNode(
        frozenset({create_variable_node(1), create_move_node(6)})
    )
    inner_s9 = SumNode(frozenset({create_move_node(4), create_move_node(6)}))
    pattern9 = SumNode(frozenset({create_variable_node(0), inner_p9}))
    subtree9 = SumNode(frozenset({create_move_node(3), inner_s9}))
    bindings9: Bindings = {}

    # Expected binding based on sorting by str:
    # Pattern sorted: [str(inner_p9), 'Var(0)'] -> ['[6|Var(1)]', 'Var(0)']
    # Subtree sorted: [str(create_move_node(3)), str(inner_s9)] -> ['3', '[4|6]']
    # Match '[6|Var(1)]' -> '[4|6]' (requires inner unification)
    #   Inner unification: p=['6', 'Var(1)'], s=['4', '6'] -> Match '6'->'6', Match 'Var(1)'->'4'. Bindings: {1: create_move_node(4)}
    # Match 'Var(0)' -> '3'. Bindings: {0: create_move_node(3)}
    # Combine bindings: {0: create_move_node(3), 1: create_move_node(4)}
    expected_bindings9 = {0: create_move_node(3), 1: create_move_node(4)}

    assert unify(pattern9, subtree9, bindings9), (
        "Nested SumNode unification should succeed"
    )
    assert bindings9 == expected_bindings9, (
        f"Bindings (9) mismatch: Expected {expected_bindings9}, got {bindings9}"
    )

    # Test Case 10: Variable bound to different type
    pattern10 = SumNode(
        frozenset({create_variable_node(0), create_move_node(1)})
    )
    subtree10 = SumNode(
        frozenset(
            {
                RepeatNode(create_variable_node(0), CountValue(4)),
                create_move_node(1),
            }
        )
    )  # Bind Var(0) to a RepeatNode
    bindings10: Bindings = {}
    assert unify(pattern10, subtree10, bindings10), (
        "Variable binding to CoordValue should succeed"
    )
    assert bindings10 == {
        0: RepeatNode(create_variable_node(0), CountValue(4))
    }, (
        f"Expected Var(0) to bind to {RepeatNode(create_variable_node(0), CountValue(4))}, got {bindings10}"
    )

    print("\nAll test_unify_sum_deterministic tests passed!")


def test_expand_nested_node():
    """
    Tests the expand_nested_node function, which expands a NestedNode by recursively
    applying a template from a symbol table to a terminal node for a specified count.
    """
    # Common node definitions
    move0 = PrimitiveNode(MoveValue(0))  # Template prefix
    move1 = PrimitiveNode(MoveValue(1))  # Terminal node
    move2 = PrimitiveNode(MoveValue(2))  # SumNode alternative
    move3 = PrimitiveNode(MoveValue(3))  # Template suffix
    var0 = VariableNode(VariableValue(0))  # Variable for substitution

    # Test Case 1: Simple ProductNode Template with SumNode
    # Template: ProductNode((move3, SumNode(frozenset({move0, var0}))))
    template1 = ProductNode((move3, SumNode(frozenset({move0, var0}))))
    symbols1 = (template1,)

    # Count=1: ProductNode((move3, SumNode(frozenset({move0, move2}))))
    nested1 = NestedNode(IndexValue(0), move2, CountValue(1))
    expected1 = ProductNode((move3, SumNode(frozenset({move0, move2}))))
    result1 = expand_nested_node(nested1, symbols1)
    assert result1 == expected1, (
        f"Test Case 1 (count=1) Failed: Expected {expected1}, got {result1}"
    )

    # Count=2: ProductNode((move3, SumNode(frozenset({move0, expected1}))))
    nested2 = NestedNode(IndexValue(0), move2, CountValue(2))
    expected2 = ProductNode((move3, SumNode(frozenset({move0, expected1}))))
    result2 = expand_nested_node(nested2, symbols1)
    assert result2 == expected2, (
        f"Test Case 1 (count=2) Failed: Expected {expected2}, got {result2}"
    )

    # Count=3:
    nested3 = NestedNode(IndexValue(0), move2, CountValue(3))
    expected3 = ProductNode((move3, SumNode(frozenset({move0, expected2}))))
    result3 = expand_nested_node(nested3, symbols1)
    assert result3 == expected3, (
        f"Test Case 1 (count=3) Failed: Expected {expected3}, got {result3}"
    )

    # Test Case 2: Complex Template with SumNode
    # Template: ProductNode((MoveValue(0), SumNode({Var(0), MoveValue(2)}), MoveValue(3)))
    sum_template = SumNode(frozenset({var0, move3}))
    template2 = ProductNode((move0, sum_template, move3))
    symbols2 = (template2,)

    # Count=1: ProductNode((MoveValue(0), SumNode({MoveValue(1), MoveValue(3)}), MoveValue(3)))
    nested1_2 = NestedNode(IndexValue(0), move1, CountValue(1))
    expected1_2 = ProductNode(
        (move0, SumNode(frozenset({move1, move3})), move3)
    )
    result1_2 = expand_nested_node(nested1_2, symbols2)
    assert result1_2 == expected1_2, (
        f"Test Case 2 (count=1) Failed: Expected {expected1_2}, got {result1_2}"
    )

    # Count=2: ProductNode((MoveValue(0), SumNode({ProductNode((MoveValue(0), SumNode({MoveValue(1), MoveValue(3)}), MoveValue(3))), MoveValue(2)}), MoveValue(3)))
    nested2_2 = NestedNode(IndexValue(0), move1, CountValue(2))
    inner_sum = SumNode(frozenset({move1, move3}))
    inner_product = ProductNode((move0, inner_sum, move3))
    expected2_2 = ProductNode(
        (move0, SumNode(frozenset({inner_product, move3})), move3)
    )
    result2_2 = expand_nested_node(nested2_2, symbols2)
    assert result2_2 == expected2_2, (
        f"Test Case 2 (count=2) Failed: Expected {expected2_2}, got {result2_2}"
    )

    print("Test expand_nested_node - Passed")


def test_factor_by_existing_symbols():
    """Tests the factor_by_existing_symbols function for replacing matching patterns with SymbolNodes."""
    # Test Case 1: Basic Pattern Replacement
    pattern = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    tree = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    symbols = (pattern,)
    result = factor_by_existing_symbols(tree, symbols)
    expected = RootNode(
        SymbolNode(IndexValue(0), ()),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert result == expected, (
        "Test Case 1 Failed: Should replace pattern with SymbolNode"
    )
    print("Test Case 1: Basic Pattern Replacement - Passed")

    # Test Case 2: No Matching Pattern
    tree = RootNode(
        PrimitiveNode(MoveValue(1)),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    symbols = (pattern,)
    result = factor_by_existing_symbols(tree, symbols)
    assert result == tree, (
        "Test Case 2 Failed: Should return unchanged tree when no match"
    )
    print("Test Case 2: No Matching Pattern - Passed")

    # Test Case 3: Nested Pattern
    nested_tree = ProductNode((pattern, PrimitiveNode(MoveValue(4))))
    result = factor_by_existing_symbols(nested_tree, symbols)
    expected = ProductNode(
        (
            SymbolNode(IndexValue(0), ()),
            PrimitiveNode(MoveValue(4)),
        )
    )
    assert result == expected, (
        "Test Case 3 Failed: Should replace nested pattern"
    )
    print("Test Case 3: Nested Pattern - Passed")

    # Test Case 4: Multiple Matches
    multi_tree = ProductNode((pattern, pattern))
    result = factor_by_existing_symbols(multi_tree, symbols)
    symbol_node = SymbolNode(IndexValue(0), ())
    expected = RepeatNode(symbol_node, CountValue(2))
    assert result == expected, (
        "Test Case 4 Failed: Should replace multiple occurrences"
    )
    print("Test Case 4: Multiple Matches - Passed")

    # Test Case 5: Empty Symbol Table
    result = factor_by_existing_symbols(tree, ())
    assert result == tree, (
        "Test Case 5 Failed: Should return unchanged tree with empty symbols"
    )
    print("Test Case 5: Empty Symbol Table - Passed")

    print("Test factor_by_existing_symbols - Passed")


def test_substitute():
    # Invariant when non variables
    template_1 = ProductNode(
        children=(
            SymbolNode(
                index=IndexValue(16),
                parameters=(CoordValue(Coord(5, 1)),),
            ),
            RootNode(
                node=NoneValue(),
                position=CoordValue(Coord(5, 1)),
                colors=PaletteValue(frozenset({1})),
            ),
        )
    )

    result = substitute_variables(template_1, ())
    assert result == ProductNode(
        children=(
            SymbolNode(
                index=IndexValue(16),
                parameters=(CoordValue(Coord(5, 1)),),
            ),
            RootNode(
                node=NoneValue(),
                position=CoordValue(Coord(5, 1)),
                colors=PaletteValue(frozenset({1})),
            ),
        )
    )

    template = RootNode(
        node=PrimitiveNode(MoveValue(2)),
        position=VariableNode(index=VariableValue(0)),
        colors=PaletteValue(frozenset({4})),
    )
    params = (CoordValue(Coord(5, 1)),)
    expanded = substitute_variables(template, params)

    # --- assertions --------------------------------------------------------
    # 1. The position attribute should now be a concrete coordinate
    assert isinstance(expanded, RootNode)
    assert isinstance(expanded.position, CoordValue)
    assert expanded.position.value == (5, 1)

    # 2. No VariableNode should survive anywhere in the subtree
    def _contains_var(node):
        if isinstance(node, VariableNode):
            return True
        return any(
            _contains_var(child)
            for child in getattr(node, "children", lambda: [])()
        )

    assert not _contains_var(expanded), "VariableNode leaked from substitution"


def test_remap_symbol_indices():
    """Tests the remap_symbol_indices function for updating SymbolNode indices."""
    # Test Case 1: Single Symbol Remap
    tree = SymbolNode(IndexValue(0), ())
    mapping = [1]
    result = remap_symbol_indices(tree, mapping, 0)
    expected = SymbolNode(IndexValue(1), ())
    assert result == expected, "Test Case 1 Failed: Should remap index 0 to 1"
    print("Test Case 1: Single Symbol Remap - Passed")

    # Test Case 2: No Symbols
    tree = ProductNode(
        (PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3)))
    )
    mapping = [1, 2]
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 2 Failed: Should return unchanged tree with no symbols"
    )
    print("Test Case 2: No Symbols - Passed")

    # Test Case 3: Nested Symbols
    tree = ProductNode(
        (SymbolNode(IndexValue(0), ()), SymbolNode(IndexValue(1), ()))
    )
    mapping = [2, 0]
    result = remap_symbol_indices(tree, mapping, 0)
    expected = ProductNode(
        (SymbolNode(IndexValue(2), ()), SymbolNode(IndexValue(0), ()))
    )
    assert result == expected, (
        "Test Case 3 Failed: Should remap nested symbols correctly"
    )
    print("Test Case 3: Nested Symbols - Passed")

    # Test Case 4: Out of Bounds Index
    tree = SymbolNode(IndexValue(2), ())
    mapping = [0, 1]  # Mapping shorter than index
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 4 Failed: Should keep unchanged when index out of bounds"
    )
    print("Test Case 4: Out of Bounds Index - Passed")

    # Test Case 5: Empty Mapping
    tree = SymbolNode(IndexValue(0), ())
    mapping = []
    result = remap_symbol_indices(tree, mapping, 0)
    assert result == tree, (
        "Test Case 5 Failed: Should keep unchanged with empty mapping"
    )
    print("Test Case 5: Empty Mapping - Passed")

    print("Test remap_symbol_indices - Passed")


def test_remap_sub_symbols():
    """Tests the remap_sub_symbols function for updating SymbolNodes within a symbol."""
    # Test Case 1: Simple Symbol with Sub-Symbol
    symbol = SymbolNode(IndexValue(0), (SymbolNode(IndexValue(1), ()),))
    mapping = [2, 0]
    original_table = (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(IndexValue(2), (SymbolNode(IndexValue(0), ()),))
    assert result == expected, (
        "Test Case 1 Failed: Should remap sub-symbol index"
    )
    print("Test Case 1: Simple Symbol with Sub-Symbol - Passed")

    # Test Case 2: No Sub-Symbols
    symbol = SymbolNode(IndexValue(0), (PrimitiveNode(MoveValue(1)),))
    mapping = [1]
    original_table = (PrimitiveNode(MoveValue(2)),)
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(IndexValue(1), (PrimitiveNode(MoveValue(1)),))
    assert result == expected, (
        "Test Case 2 Failed: Should remap outer symbol index"
    )
    print("Test Case 2: No Sub-Symbols - Passed")

    # Test Case 3: Multiple Sub-Symbols
    symbol = SymbolNode(
        IndexValue(0),
        (SymbolNode(IndexValue(0), ()), SymbolNode(IndexValue(1), ())),
    )
    mapping = [1, 0]
    original_table = (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    result = remap_sub_symbols(symbol, mapping, original_table)
    expected = SymbolNode(
        IndexValue(1),
        (SymbolNode(IndexValue(1), ()), SymbolNode(IndexValue(0), ())),
    )
    assert result == expected, (
        "Test Case 3 Failed: Should remap multiple sub-symbols"
    )
    print("Test Case 3: Multiple Sub-Symbols - Passed")

    # Test Case 4: Empty Mapping
    symbol = SymbolNode(IndexValue(0), (SymbolNode(IndexValue(0), ()),))
    mapping = []
    original_table = (PrimitiveNode(MoveValue(1)),)
    result = remap_sub_symbols(symbol, mapping, original_table)
    assert result == symbol, (
        "Test Case 4 Failed: Should return unchanged with empty mapping"
    )
    print("Test Case 4: Empty Mapping - Passed")

    print("Test remap_sub_symbols - Passed")


def test_merge_symbol_tables():
    """Tests the merge_symbol_tables function for combining symbol tables."""
    # Test Case 1: Merging Identical Tables
    symbol = ProductNode(
        (PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)))
    )
    table1 = (symbol,)
    table2 = (symbol,)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == (symbol,), "Test Case 1 Failed: Unified table incorrect"
    assert mappings == [[0], [0]], (
        "Test Case 1 Failed: Mappings should point to same index"
    )
    print("Test Case 1: Merging Identical Tables - Passed")

    # Test Case 2: Merging Distinct Tables
    symbol1 = PrimitiveNode(MoveValue(1))
    symbol2 = PrimitiveNode(MoveValue(2))
    table1 = (symbol1,)
    table2 = (symbol2,)
    unified, mappings = merge_symbol_tables([table1, table2])
    # Both symbols are independent, so topological sort can return either order
    assert set(unified) == {symbol1, symbol2}, (
        "Test Case 2 Failed: Unified table should contain both symbols"
    )
    # Verify mappings point to correct indices regardless of order
    assert unified[mappings[0][0]] == symbol1, "Test Case 2 Failed: Mapping for table1 incorrect"
    assert unified[mappings[1][0]] == symbol2, "Test Case 2 Failed: Mapping for table2 incorrect"
    print("Test Case 2: Merging Distinct Tables - Passed")

    # Test Case 3: Merging with Nested Symbols
    nested_symbol = RepeatNode(SymbolNode(IndexValue(0), ()), CountValue(3))
    symbol_def = PrimitiveNode(MoveValue(1))
    table1 = (symbol_def, nested_symbol)
    table2 = (symbol_def,)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == table1, "Test Case 3 Failed: Unified table incorrect"
    assert mappings == [[0, 1], [0]], "Test Case 3 Failed: Mappings incorrect"
    print("Test Case 3: Merging with Nested Symbols - Passed")

    # Test Case 4: Merging with Nested Symbols and Variables
    resolved = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(4)), PrimitiveNode(MoveValue(5)))),
        CountValue(3),
    )
    symbol1 = RepeatNode(
        ProductNode((PrimitiveNode(MoveValue(4)), PrimitiveNode(MoveValue(5)))),
        VariableNode(VariableValue(0)),
    )
    symbol2 = SymbolNode(IndexValue(0), (CountValue(3),))
    table1 = (resolved,)
    table2 = (symbol1, symbol2)
    unified, mappings = merge_symbol_tables([table1, table2])
    assert unified == table2, "Test Case 3 Failed: Unified table incorrect"
    assert mappings == [[1], [0, 1]], "Test Case 3 Failed: Mappings incorrect"
    print("Test Case 4: Merging with Nested Symbols and Variables - Passed")

    # Test Case 4: Empty Tables
    unified, mappings = merge_symbol_tables([(), ()])
    assert unified == (), (
        "Test Case 4 Failed: Should return empty unified table"
    )
    assert mappings == [[], []], (
        "Test Case 4 Failed: Should return empty mappings"
    )
    print("Test Case 4: Empty Tables - Passed")

    print("Test merge_symbol_tables - Passed")


def test_symbolize_together():
    """Tests the symbolize_together function for unifying and re-symbolizing trees."""
    # Test Case 1: Identical Trees with Empty Symbol Tables
    pattern = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    trees = (tree1, tree2)
    symbol_tables = [(), ()]
    final_trees, final_symbols = symbolize_together(trees, symbol_tables)
    expected_symbol = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    expected_trees = (
        RootNode(
            SymbolNode(IndexValue(0), tuple()),
            CoordValue(Coord(0, 0)),
            PaletteValue(value=frozenset({1})),
        ),
        RootNode(
            SymbolNode(IndexValue(0), tuple()),
            CoordValue(Coord(1, 1)),
            PaletteValue(value=frozenset({2})),
        ),
    )
    expected_symbols = (expected_symbol,)
    assert final_trees == expected_trees, (
        f"Test Case 1 Failed: Trees incorrect, got {final_trees}"
    )
    assert final_symbols == expected_symbols, (
        "Test Case 1 Failed: Symbols incorrect"
    )
    print("Test Case 1: Identical Trees with Empty Symbol Tables - Passed")

    # Test Case 2: Trees with Pre-existing Symbols
    symbol_def = PrimitiveNode(MoveValue(1))
    tree1 = SymbolNode(IndexValue(0), ())
    tree2 = SymbolNode(IndexValue(0), ())
    symbol_tables = [(symbol_def,), (symbol_def,)]
    final_trees, final_symbols = symbolize_together(
        (tree1, tree2), symbol_tables
    )
    assert final_trees == (tree1, tree2), (
        "Test Case 2 Failed: Trees should remain unchanged"
    )
    assert final_symbols == (symbol_def,), (
        "Test Case 2 Failed: Symbols should be unified"
    )
    print("Test Case 2: Trees with Pre-existing Symbols - Passed")

    # Test Case 3: Empty Input
    final_trees, final_symbols = symbolize_together((), [])
    assert final_trees == (), (
        "Test Case 3 Failed: Empty trees should return empty"
    )
    assert final_symbols == (), (
        "Test Case 3 Failed: Empty symbols should return empty"
    )
    print("Test Case 3: Empty Input - Passed")

    # Test Case 4: Integration Test with Resolution
    pattern = ProductNode(
        (
            PrimitiveNode(MoveValue(2)),
            PrimitiveNode(MoveValue(3)),
            PrimitiveNode(MoveValue(4)),
            PrimitiveNode(MoveValue(4)),
        )
    )
    tree1 = RootNode(
        pattern, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    tree2 = RootNode(
        pattern, CoordValue(Coord(1, 1)), PaletteValue(frozenset({2}))
    )
    trees = (tree1, tree2)
    symbol_tables = [(), ()]  # Empty symbol tables to start fresh
    final_trees, final_symbols = symbolize_together(trees, symbol_tables)
    resolved_trees = tuple(
        resolve_symbols(tree, final_symbols) for tree in final_trees
    )
    expected_resolved = (
        tree1,
        tree2,
    )  # Expect the original trees after resolution
    assert resolved_trees == expected_resolved, (
        "Test Case 4 Failed: Resolved trees should match originals"
    )
    print("Test Case 4: Integration Test with Resolution - Passed")

    print("Test symbolize_together - Passed")


def run_tests():
    """Runs simple tests to verify KolmogorovTree functionality."""
    # Test 1: Basic node creation and bit length
    move_right = PrimitiveNode(MoveValue(2))
    assert (
        move_right.bit_length()
        == BitLength.NODE_TYPE + MoveValue(2).bit_length()
    ), (
        f"PrimitiveNode bit length should be {BitLength.NODE_TYPE + MoveValue(2).bit_length()} ( {BitLength.NODE_TYPE} + {MoveValue(2).bit_length()})"
    )
    product = ProductNode((move_right, move_right))
    assert (
        product.bit_length()
        == BitLength.NODE_TYPE + 2 + 2 * move_right.bit_length()
    ), (
        f"ProductNode bit length should be {BitLength.NODE_TYPE + 2 + 2 * move_right.bit_length()} ({BitLength.NODE_TYPE} + 2 + 2 * {move_right.bit_length()})"
    )
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode(frozenset((move_right, move_down)))
    assert (
        sum_node.bit_length()
        == BitLength.NODE_TYPE
        + 2
        + move_right.bit_length()
        + move_down.bit_length()
    ), (
        f"SumNode bit length should be {BitLength.NODE_TYPE + 2 + move_right.bit_length() + move_down.bit_length()} ({BitLength.NODE_TYPE} + 2 + {move_right.bit_length()} + {move_down.bit_length()})"
    )
    repeat = RepeatNode(move_right, CountValue(3))
    assert (
        repeat.bit_length()
        == BitLength.NODE_TYPE + BitLength.COUNT + move_right.bit_length()
    ), (
        f"RepeatNode bit length should be {BitLength.NODE_TYPE + BitLength.COUNT + move_right.bit_length()} ({BitLength.NODE_TYPE} + {move_right.bit_length()} + {BitLength.COUNT})"
    )
    nested = NestedNode(IndexValue(0), repeat, CountValue(3))
    assert (
        nested.bit_length()
        == BitLength.NODE_TYPE
        + BitLength.COUNT
        + BitLength.INDEX
        + repeat.bit_length()
    ), f"RepeatNode bit length should be {
        BitLength.NODE_TYPE
        + BitLength.INDEX
        + BitLength.COUNT
        + sum_node.bit_length()
        + repeat.bit_length()
    } ({BitLength.NODE_TYPE} + {BitLength.INDEX} + {repeat.bit_length()} + {
        BitLength.COUNT
    })"
    symbol = SymbolNode(IndexValue(0), ())
    assert symbol.bit_length() == BitLength.NODE_TYPE + BitLength.INDEX, (
        f"SymbolNode bit length should be {BitLength.NODE_TYPE + BitLength.INDEX} ({BitLength.NODE_TYPE} + {BitLength.INDEX})"
    )
    root = RootNode(
        product, CoordValue(Coord(0, 0)), PaletteValue(frozenset({1}))
    )
    assert (
        root.bit_length()
        == BitLength.NODE_TYPE
        + product.bit_length()
        + ARCBitLength.COORD
        + ARCBitLength.COLORS
    ), (
        "RootNode bit length should be {BitLength.NODE_TYPE + product.bit_length() + ARCBitLength.COORD + ARCBitLength.COLORS} ({BitLength.NODE_TYPE} + {product.bit_length()} + {ARCBitLength.COORD} + {ARCBitLength.COLORS})"
    )
    print("Test 1: Basic node creation and bit length - Passed")

    # Test 2: String representations
    assert str(move_right) == "2", "PrimitiveNode str should be '2'"
    assert str(product) == "[2,2]", (
        f"ProductNode str should be '[2,2]', instead of {str(product)}"
    )
    assert str(sum_node) == "{2,3}", "SumNode str should be '{2,3}'"
    assert str(repeat) == "(2)*{3}", "RepeatNode str should be '(2)*{3}'"
    assert str(symbol).strip() == "s_0()", (
        f"SymbolNode str should be 's_0', got {str(symbol)}"
    )
    assert str(root) == "Root([2,2], (0, 0), set{1})", (
        f"RootNode str should be 'Root((2.2), (0, 0), {1})'. Got {str(root)} instead"
    )
    print("Test 2: String representations - Passed")

    # Test 3: Operator overloads
    assert (move_right | move_down) == SumNode(
        frozenset((move_right, move_down))
    ), "Operator | failed"
    assert (move_right & move_down) == ProductNode((move_right, move_down)), (
        "Operator & failed"
    )
    assert (move_right + move_down) == ProductNode((move_right, move_down)), (
        "Operator + failed"
    )
    assert (move_right * 3) == RepeatNode(move_right, CountValue(3)), (
        "Operator * failed"
    )
    assert ((move_right * 2) * 3) == RepeatNode(move_right, CountValue(6)), (
        "Nested * failed"
    )
    print("Test 3: Operator overloads - Passed")

    # Test 4: Symbol resolution
    symbol_def = ProductNode((move_right, move_down, move_right))
    symbols: tuple[KNode[MoveValue], ...] = (symbol_def,)
    symbol_node = SymbolNode(IndexValue(0), ())
    root_with_symbol = RootNode(
        symbol_node,
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    resolved = resolve_symbols(root_with_symbol, symbols)
    expected_resolved = RootNode(
        ProductNode((move_right, move_down, move_right)),
        CoordValue(Coord(0, 0)),
        PaletteValue(frozenset({1})),
    )
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

    test_encode_run_length()
    test_construct_product_node()
    test_shift()
    test_get_iterator()
    test_arity()

    test_find_repeating_pattern()
    test_factorize_tuple()
    test_is_abstraction()
    test_resolve_symbols()
    test_find_symbol_candidates()
    test_matching()

    test_unify_sum()

    test_extract_nested_sum_template()
    test_node_to_symbolize()
    test_extract_nested_product_template()
    test_nested_collection_to_nested_node()
    test_expand_nested_node()
    test_substitute()
    test_factor_by_existing_symbols()
    test_symbolize_pattern()
    test_remap_symbol_indices()
    test_remap_sub_symbols()
    test_merge_symbol_tables()
    test_symbolize_together()


if __name__ == "__main__":
    run_tests()
    print("All tests passed successfully!")
