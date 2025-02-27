"""
Kolmogorov Tree: A representation language for non-deterministic programs and patterns.

This module implements a tree structure for representing non-deterministic programs,
particularly focused on 2D grid movements and shape descriptions. The representation
aims to approximate Kolmogorov complexity through minimum description length.

The tree structure consists of:
- ProductNodes for deterministic sequences
- SumNodes for non-deterministic branching
- RepeatNodes for repetition extraction
- SymbolNodes for pattern abstraction and reuse
- Variable binding for lambda abstraction

Key Features:
- Computable bit-length metrics for complexity approximation
- Pattern extraction and memorization via symbol table
- Support for 8-directional grid movements
- Lambda abstraction through variable binding
- Nested pattern detection and reuse

The tree can be used to:
1. Represent shapes as non-deterministic programs
2. Extract common patterns across multiple programs
3. Measure and optimize program complexity
4. Enable search over program space guided by complexity

Example Usage:
```python
# Create a simple pattern
pattern = create_moves_sequence("2323")  # Right-Down-Right-Down
# Repeat it 3 times
repeated = RepeatNode(pattern, 3)
# Add non-deterministic branching
program = SumNode([repeated, create_rect(3, 3)])
# Create full tree with starting position and colors
tree = KolmogorovTree(RootNode((0,0), {1}, program))
```
"""


from typing import Generic, TypeVar, Callable, Iterable, Any
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import Counter
import math

# Assuming RoseTree is defined elsewhere; import it
from tree import RoseTree, Rose

# Type variable for generic programming
T = TypeVar('T')

# Bit length definitions
class BitLength(IntEnum):
    """Defines bit lengths for components in the Kolmogorov Tree."""
    NODE_TYPE = 3  # 3 bits for up to 8 node types
    PRIMITIVE = 4  # 4 bits for primitives (e.g., directions 0-7, colors 0-9)
    COUNT = 5      # 5 bits for repeat counts (0-31), suitable for ARC grid sizes
    COORD = 10     # 10 bits for coordinates (5 bits per x/y, for 0-31)
    INDEX = 4      # 4 bits for symbol indices (up to 16 symbols)
    VAR = 2        # 2 bits for variable indices (up to 4 variables per symbol)

# Base interface for values with bit lengths
@dataclass(frozen=True)
class BitLengthAware(ABC):
    """Interface for values that can report their bit lengths."""

    @abstractmethod
    def bit_length(self) -> int:
        """Returns the bit length of this value."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns the string representation of this value."""
        pass

# Base node class
@dataclass(frozen=True)
class KNode(Generic[T], ABC):
    def __len__(self) -> int:
        return BitLength.NODE_TYPE

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    @abstractmethod
    def bit_length(self) -> int:
        pass

    def __or__(self, other: 'KNode[T]') -> 'SumNode[T]':
        """Overloads | for alternatives, unpacking SumNodes."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        children = []
        if isinstance(self, SumNode):
            children.extend(self.children)
        else:
            children.append(self)
        if isinstance(other, SumNode):
            children.extend(other.children)
        else:
            children.append(other)
        return SumNode(tuple(children))

    def __and__(self, other: 'KNode[T]') -> 'ProductNode[T]':
        """Overloads & for sequences, unpacking ProductNodes."""
        if not isinstance(other, KNode):
            raise TypeError("Operand must be a KNode")
        children = []
        if isinstance(self, ProductNode):
            children.extend(self.children)
        else:
            children.append(self)
        if isinstance(other, ProductNode):
            children.extend(other.children)
        else:
            children.append(other)
        return ProductNode(tuple(children))

    def __add__(self, other: 'KNode[T]') -> 'ProductNode[T]':
        """Overloads + for concatenation, unpacking ProductNodes."""
        return self.__and__(other)  # Same behavior as &

    def __mul__(self, count: int) -> 'RepeatNode[T]':
        """Overloads * for repetition, multiplying count if already a RepeatNode."""
        if not isinstance(count, int):
            raise TypeError("Count must be an integer")
        if count < 0:
            raise ValueError("Count must be non-negative")
        if isinstance(self, RepeatNode) and isinstance(self.count, int):
            # Multiply existing count to optimize complexity
            return RepeatNode(self.node, self.count * count)
        return RepeatNode(self, count)

# Primitive value types
@dataclass(frozen=True)
class MoveValue(BitLengthAware):
    """Represents a single directional move (0-7 for 8-connectivity)."""
    direction: int

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE  # 4 bits

    def __str__(self) -> str:
        return str(self.direction)

@dataclass(frozen=True)
class MovesValue(BitLengthAware):
    """Represents a sequence of directional moves."""
    directions: tuple[int]

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE * len(self.directions)

    def __str__(self) -> str:
        return "".join(map(str, self.directions))

@dataclass(frozen=True)
class ColorValue(BitLengthAware):
    """Represents a color value (0-9 in ARC AGI)."""
    value: int

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE  # 4 bits

    def __str__(self) -> str:
        return f"C{self.value}"

@dataclass(frozen=True)
class CoordValue(BitLengthAware):
    """Represents a 2D coordinate pair."""
    x: int
    y: int

    def bit_length(self) -> int:
        return BitLength.COORD  # 10 bits (5 per coordinate)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

# Node types for program representation
@dataclass(frozen=True)
class PrimitiveNode(KNode[T]):
    """Leaf node holding a primitive value."""
    value: BitLengthAware

    def bit_length(self) -> int:
        return super().__len__() + self.value.bit_length()

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class VariableNode(KNode[T]):
    """Represents a variable placeholder within a symbol."""
    index: int  # Variable index (0-3 with 2 bits)

    def bit_length(self) -> int:
        return super().__len__() + BitLength.VAR  # 3 + 2 bits

    def __str__(self) -> str:
        return f"Var({self.index})"

@dataclass(frozen=True)
class SequenceNode(KNode[T], ABC):
    """Abstract base class for nodes with multiple children."""
    children: tuple[KNode[T], ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Converts children to tuple if provided as a list."""
        if isinstance(self.children, list):
            object.__setattr__(self, 'children', tuple(self.children))

@dataclass(frozen=True)
class ProductNode(SequenceNode[T]):
    """Represents a sequence of actions (AND operation)."""
    def bit_length(self) -> int:
        return super().__len__() + sum(child.bit_length() for child in self.children)

    def __str__(self) -> str:
        return "".join(str(child) for child in self.children)

@dataclass(frozen=True)
class SumNode(SequenceNode[T]):
    """Represents a choice among alternatives (OR operation)."""
    def bit_length(self) -> int:
        select_bits = math.ceil(math.log2(len(self.children) + 1)) if self.children else 0
        return super().__len__() + select_bits + sum(child.bit_length() for child in self.children)

    def __str__(self) -> str:
        return "[" + "|".join(str(child) for child in self.children) + "]"

@dataclass(frozen=True)
class RepeatNode(KNode[T]):
    """Represents repetition of a node a specified number of times."""
    node: KNode[T]
    count: int | VariableNode  # Count can be fixed or parameterized

    def bit_length(self) -> int:
        count_len = BitLength.COUNT if isinstance(self.count, int) else self.count.bit_length()
        return super().__len__() + self.node.bit_length() + count_len

    def __str__(self) -> str:
        return f"({str(self.node)})*{{{self.count}}}"

@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """Represents an abstraction or reusable pattern."""
    index: int  # Index in the symbol table (0-15 with 4 bits)
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)
    referenced_length: int = 0  # Bit length of the referenced pattern

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().__len__() + BitLength.INDEX + params_len

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}"

@dataclass(frozen=True)
class RootNode(KNode[T]):
    """Root node defining the program's starting context."""
    position: tuple[int, int] | VariableNode  # Starting position
    colors: set[int] | VariableNode  # Colors used in the shape
    node: KNode[T] | None = None  # Main program node

    def bit_length(self) -> int:
        pos_len = BitLength.COORD if isinstance(self.position, tuple) else self.position.bit_length()
        colors_len = (BitLength.PRIMITIVE * len(self.colors)
                     if isinstance(self.colors, set) else self.colors.bit_length())
        node_len = self.node.bit_length() if self.node else 0
        return super().__len__() + pos_len + colors_len + node_len

    def __str__(self) -> str:
        pos_str = str(self.position) if isinstance(self.position, tuple) else str(self.position)
        colors_str = str(self.colors) if isinstance(self.colors, set) else str(self.colors)
        node_str = str(self.node) if self.node else ""
        return f"Root({pos_str}, {colors_str}, {node_str})"

# Main tree class
@dataclass
class KolmogorovTree:
    """Represents a complete program with a root node and symbol table."""
    root: KNode
    symbols: list[KNode] = field(default_factory=list)

    def bit_length(self) -> int:
        """Total bit length includes root and symbol definitions."""
        return self.root.bit_length() + sum(symbol.bit_length() for symbol in self.symbols)

    def resolve_symbols(self) -> KNode:
        """Expands all symbols into a fully resolved tree."""
        return resolve_symbols(self.root, self.symbols)

    def __str__(self) -> str:
        symbols_str = "\n".join(f"s_{i}: {str(s)}" for i, s in enumerate(self.symbols))
        return f"Symbols:\n{symbols_str}\n\nProgram:\n{str(self.root)}"

    def kolmogorov_complexity(self) -> int:
        """Approximates Kolmogorov complexity as the total bit length."""
        return self.bit_length()

# Symbol resolution and pattern finding
def resolve_symbols(node: KNode, symbols: list[KNode]) -> KNode:
    """Resolves symbolic references recursively using the symbol table."""
    def replace_variables(template: KNode, params: tuple[Any, ...]) -> KNode:
        if isinstance(template, VariableNode) and template.index < len(params):
            param = params[template.index]
            return param if isinstance(param, KNode) else PrimitiveNode(param)

        if isinstance(template, ProductNode):
            return ProductNode([replace_variables(child, params) for child in template.children])
        if isinstance(template, SumNode):
            return SumNode([replace_variables(child, params) for child in template.children])
        if isinstance(template, RepeatNode):
            new_node = replace_variables(template.node, params)
            new_count = (replace_variables(template.count, params)
                        if isinstance(template.count, KNode) else template.count)
            return RepeatNode(new_node, new_count)
        if isinstance(template, SymbolNode):
            new_params = tuple(replace_variables(p, params) if isinstance(p, KNode) else p
                              for p in template.parameters)
            return SymbolNode(template.index, new_params, template.referenced_length)
        if isinstance(template, RootNode):
            new_pos = (replace_variables(template.position, params)
                      if isinstance(template.position, KNode) else template.position)
            new_colors = (replace_variables(template.colors, params)
                         if isinstance(template.colors, KNode) else template.colors)
            new_node = replace_variables(template.node, params) if template.node else None
            return RootNode(new_pos, new_colors, new_node)
        return template

    if isinstance(node, SymbolNode) and 0 <= node.index < len(symbols):
        return replace_variables(symbols[node.index], node.parameters)
    if isinstance(node, ProductNode):
        return ProductNode([resolve_symbols(child, symbols) for child in node.children])
    if isinstance(node, SumNode):
        return SumNode([resolve_symbols(child, symbols) for child in node.children])
    if isinstance(node, RepeatNode):
        new_node = resolve_symbols(node.node, symbols)
        new_count = (resolve_symbols(node.count, symbols)
                    if isinstance(node.count, KNode) else node.count)
        return RepeatNode(new_node, new_count)
    if isinstance(node, RootNode):
        new_pos = (resolve_symbols(node.position, symbols)
                  if isinstance(node.position, KNode) else node.position)
        new_colors = (resolve_symbols(node.colors, symbols)
                     if isinstance(node.colors, KNode) else node.colors)
        new_node = resolve_symbols(node.node, symbols) if node.node else None
        return RootNode(new_pos, new_colors, new_node)
    return node

def find_common_patterns(trees: list[KolmogorovTree], min_occurrences=2, max_patterns=10) -> list[KNode]:
    """Identifies frequent subtrees across multiple trees for symbolization."""
    all_subtrees = []

    def collect_subtrees(node, collection):
        collection.append(node)
        if isinstance(node, (ProductNode, SumNode)):
            for child in node.children:
                collect_subtrees(child, collection)
        elif isinstance(node, RepeatNode):
            collect_subtrees(node.node, collection)
        elif isinstance(node, RootNode) and node.node:
            collect_subtrees(node.node, collection)
        elif isinstance(node, SymbolNode):
            for param in node.parameters:
                if isinstance(param, KNode):
                    collect_subtrees(param, collection)

    for tree in trees:
        collect_subtrees(tree.root, all_subtrees)
        for symbol in tree.symbols:
            collect_subtrees(symbol, all_subtrees)

    counter = Counter(str(node) for node in all_subtrees)
    common_nodes = []
    for node in all_subtrees:
        node_str = str(node)
        if (counter[node_str] >= min_occurrences and
            node_str not in [str(n) for n in common_nodes]):
            common_nodes.append(node)

    common_nodes.sort(key=lambda n: (-counter[str(n)], -n.bit_length()))
    return common_nodes[:max_patterns]

def symbolize(trees: list[KolmogorovTree]) -> KolmogorovTree:
    """Creates a new tree with common patterns replaced by symbols."""
    common_patterns = find_common_patterns(trees)
    symbols = list(common_patterns)

    def process_node(node):
        for i, pattern in enumerate(common_patterns):
            if str(node) == str(pattern):
                return SymbolNode(i, (), pattern.bit_length())
        if isinstance(node, ProductNode):
            return ProductNode([process_node(child) for child in node.children])
        if isinstance(node, SumNode):
            return SumNode([process_node(child) for child in node.children])
        if isinstance(node, RepeatNode):
            return RepeatNode(process_node(node.node), node.count)
        if isinstance(node, RootNode):
            new_node = process_node(node.node) if node.node else None
            return RootNode(node.position, node.colors, new_node)
        return node

    if not trees:
        return KolmogorovTree(ProductNode([]))
    new_root = process_node(trees[0].root)
    return KolmogorovTree(new_root, symbols)

# Factory functions for common patterns
def create_move_node(direction: int) -> PrimitiveNode:
    """Creates a node for a directional move."""
    return PrimitiveNode(MoveValue(direction))

def create_color_node(color: int) -> PrimitiveNode:
    """Creates a node for a color."""
    return PrimitiveNode(ColorValue(color))

def create_coord_node(x: int, y: int) -> PrimitiveNode:
    """Creates a node for a coordinate."""
    return PrimitiveNode(CoordValue(x, y))

def create_moves_sequence(directions: str) -> KNode:
    """Creates a sequence of moves from a direction string."""
    if not directions:
        return ProductNode([])
    moves = [create_move_node(int(d)) for d in directions]
    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len
                if all(moves[i * pattern_len:(i + 1) * pattern_len] == pattern
                       for i in range(repeat_count)) and repeat_count > 1:
                    return RepeatNode(ProductNode(pattern), repeat_count)
    return ProductNode(moves)

def create_rect(height: int, width: int) -> KNode:
    """Creates a node for a rectangle shape."""
    if height < 2 or width < 2:
        return ProductNode([])
    first_row = "2" * (width - 1)  # Move right
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"  # Alternate left/right
        other_rows += "3" + direction * (width - 1)  # Down then horizontal
    return create_moves_sequence(first_row + other_rows)

# Example usage@
def example_usage():
    """Demonstrates creating and evaluating a KolmogorovTree."""
    move_right = create_move_node(2)
    move_down = create_move_node(3)
    sequence = ProductNode([move_right, move_right, move_down])
    repeat = RepeatNode(ProductNode([move_right, move_down]), 3)
    alternative = SumNode([sequence, repeat])
    root = RootNode((0, 0), {1}, alternative)
    tree = KolmogorovTree(root)
    print(f"Kolmogorov complexity: {tree.kolmogorov_complexity()} bits")
    return tree

def run_tests():
    """Runs simple tests to verify KolmogorovTree functionality."""
    # Test 1: Basic node creation and bit length
    move_right = PrimitiveNode(MoveValue(2))
    assert move_right.bit_length() == 7, "PrimitiveNode bit length should be 7 (3 + 4)"
    product = ProductNode([move_right, move_right])
    assert product.bit_length() == 17, "ProductNode bit length should be 17 (3 + 7 + 7)"
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode([move_right, move_down])
    assert sum_node.bit_length() == 19, "SumNode bit length should be 19 (3 + 2 + 7 + 7)"
    repeat = RepeatNode(move_right, 3)
    assert repeat.bit_length() == 15, "RepeatNode bit length should be 15 (3 + 7 + 5)"
    symbol = SymbolNode(0, ())
    assert symbol.bit_length() == 7, "SymbolNode bit length should be 7 (3 + 4)"
    root = RootNode((0, 0), {1}, product)
    assert root.bit_length() == 34, "RootNode bit length should be 34 (3 + 10 + 4 + 17)"
    print("Test 1: Basic node creation and bit length - Passed")

    # Test 2: String representations
    assert str(move_right) == "2", "PrimitiveNode str should be '2'"
    assert str(product) == "22", "ProductNode str should be '22'"
    assert str(sum_node) == "[2|3]", "SumNode str should be '[2|3]'"
    assert str(repeat) == "(2)*{3}", "RepeatNode str should be '(2)*{3}'"
    assert str(symbol) == "s_0", "SymbolNode str should be 's_0'"
    print("root: ", root)
    assert str(root) == "Root((0, 0), {1}, 22)", "RootNode str should be 'Root((0,0), {1}, 22)'"
    print("Test 2: String representations - Passed")

    # Test 3: Operator overloads
    assert (move_right | move_down) == SumNode([move_right, move_down]), "Operator | failed"
    assert (move_right & move_down) == ProductNode([move_right, move_down]), "Operator & failed"
    assert (move_right + move_down) == ProductNode([move_right, move_down]), "Operator + failed"
    assert (move_right * 3) == RepeatNode(move_right, 3), "Operator * failed"
    assert ((move_right * 2) * 3) == RepeatNode(move_right, 6), "Nested * failed"
    print("Test 3: Operator overloads - Passed")

    # Test 4: Symbol resolution
    symbol_def = ProductNode([move_right, move_down])
    symbols = [symbol_def]
    symbol_node = SymbolNode(0, ())
    root_with_symbol = RootNode((0, 0), {1}, ProductNode([symbol_node, move_right]))
    tree = KolmogorovTree(root_with_symbol, symbols)
    resolved = resolve_symbols(tree.root, tree.symbols)
    expected_resolved = RootNode((0, 0), {1}, ProductNode([symbol_def, move_right]))
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

if __name__ == "__main__":
    tree = example_usage()
    run_tests()
    print("All tests passed successfully!")
