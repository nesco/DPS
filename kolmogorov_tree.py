"""
Kolmogorov Tree: A representation language for non-deterministic programs and patterns.

This module implements a tree structure for representing non-deterministic programs,
for example to describe shapes using a non-deterministic program representing 2D grid movements (see syntax_tree.py). The representation
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


In a way, BitLengthAware is a type encapsulation, so the types "know" their bit length, like types that knows
their gradients in deep learning.

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


from typing import Generic, TypeVar, Callable, Iterable, Any, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import Counter
import math

# Assuming RoseTree is defined elsewhere; import it
from tree import RoseTree, Rose

# Type variable for generic programming

# Bit length definitions
class BitLength(IntEnum):
    """Defines bit lengths for generic components in the Kolmogorov Tree."""
    COUNT = 5      # 5 bits for repeat counts (0-31), specific value suitable for ARC grid sizes
    NODE_TYPE = 3  # 3 bits for up to 8 node types
    INDEX = 7      # 7 bits for symbol indices (up to 128 symbols)
    VAR = 2        # 2 bits for variable indices (up to 4 variables per symbol)

class ARCBitLength(IntEnum):
    """Defines bit lengths for components tailored for ARC AGI."""
    COORD = 10     # 10 bits for coordinates (5 bits per x/y, for 0-31)
    COLORS = 4     # 4 bits for primitives (0-9)
    DIRECTIONS = 3      # 3 bits for primitives (directions 0-7)


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

# Primitive value types
@dataclass(frozen=True)
class Primitive(BitLengthAware):
    """Base class for all primitives: bit length aware wrapping base values"""
    value: Any

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class Alphabet(Primitive):
    """Base class for all types which are program outputs. The generic type 'T' is bound to it. It offers a shifting operation"""

    @abstractmethod
    def shift(self) -> 'Alphabet':
        """Shifts the primitive value by k steps."""
        pass

    @staticmethod
    def size() -> int:
        """Size of the alphabet"""
        return 0

@dataclass(frozen=True)
class CountValue(Primitive):
    """Represent a 5-bit int"""
    value: int

    def bit_length(self) -> int:
        return BitLength.COUNT

@dataclass(frozen=True)
class VariableValue(Primitive):
    """Represents a variable as an index (0-3)."""
    value: int

    def bit_length(self) -> int:
        return BitLength.VAR

@dataclass(frozen=True)
class IndexValue(Primitive):
    """Represents an index in the lookup table (0-127)."""
    value: int

    def bit_length(self) -> int:
        return BitLength.INDEX # 7 bits

# ARC Specific primitves
@dataclass(frozen=True)
class MoveValue(Alphabet):
    """Represents a single directional move (0-7 for 8-connectivity)."""
    value: int

    def bit_length(self) -> int:
        return ARCBitLength.DIRECTIONS  # 4 bits

    def shift(self, k: int) -> 'Alphabet':
        return MoveValue((self.value + k) % 8)

    @staticmethod
    def size():
        return 8

@dataclass(frozen=True)
class PaletteValue(Primitive):
    """Represents a color value (0-9 in ARC AGI)."""
    value: set[int]

    def bit_length(self) -> int:
        return ARCBitLength.COLORS * len(self.value)  # 4 bits

@dataclass(frozen=True)
class CoordValue(Primitive):
    """Represents a 2D coordinate pair."""
    value: tuple[int, int]

    def bit_length(self) -> int:
        return ARCBitLength.COORD  # 10 bits (5 per coordinate)

T = TypeVar('T', bound=Alphabet)

# Base node class
@dataclass(frozen=True)
class KNode(Generic[T], ABC):
    def __len__(self) -> int:
        return self.bit_length()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def bit_length(self) -> int:
        return BitLength.NODE_TYPE

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
        if isinstance(self, RepeatNode) and isinstance(self.count, CountValue) and self.count.value * count < 2**BitLength.COUNT:
            # Multiply existing count to optimize complexity
            return RepeatNode(self.node, CountValue(self.count.value * count))
        return RepeatNode(self, CountValue(count))

# Node types for program representation
@dataclass(frozen=True)
class PrimitiveNode(KNode[T]):
    """Leaf node holding a primitive value."""
    value: T

    def bit_length(self) -> int:
        return super().bit_length() + self.value.bit_length()

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class VariableNode(KNode[T]):
    """Represents a variable placeholder within a symbol."""
    index: VariableValue  # Variable index (0-3 with 2 bits)

    def bit_length(self) -> int:
        return super().__len__() + self.index.bit_length()  # 3 + 2 bits

    def __str__(self) -> str:
        return f"Var({self.index})"

@dataclass(frozen=True)
class TupleNode(KNode[T], ABC):
    """Abstract base class for nodes with multiple children."""
    children: tuple[KNode[T], ...] = field(default_factory=tuple)

    def bit_length(self):
        return super().bit_length() + sum(child.bit_length() for child in self.children)

    def __post_init__(self):
        """Converts children to tuple if provided as a list."""
        if isinstance(self.children, list):
            object.__setattr__(self, 'children', tuple(self.children))

@dataclass(frozen=True)
class ProductNode(TupleNode[T]):
    """Represents a sequence of actions (AND operation)."""
    def bit_length(self) -> int:
        return super().bit_length()

    def __str__(self) -> str:
        return "".join(str(child) for child in self.children)

@dataclass(frozen=True)
class SumNode(TupleNode[T]):
    """Represents a choice among alternatives (OR operation)."""
    def bit_length(self) -> int:
        select_bits = math.ceil(math.log2(len(self.children) + 1)) if self.children else 0
        return super().bit_length() + select_bits

    def __str__(self) -> str:
        return "[" + "|".join(str(child) for child in self.children) + "]"

@dataclass(frozen=True)
class MetaNode(KNode[T], ABC):
    """Wraps a single node with additional information. node allows None to handle edge cases for map functions"""
    node: KNode[T] | None

    def bit_length(self) -> int:
        return super().bit_length() + self.node.bit_length()

@dataclass(frozen=True)
class RepeatNode(MetaNode[T]):
    """Represents repetition of a node a specified number of times."""
    count: CountValue | VariableNode  # Count can be fixed or parameterized

    def bit_length(self) -> int:
        count_len = self.count.bit_length()
        return super().bit_length() + count_len

    def __str__(self) -> str:
        return f"({str(self.node)})*{{{self.count}}}"

@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """Represents an abstraction or reusable pattern."""
    index: IndexValue  # Index in the symbol table
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_len

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}"

    def __post_init__(self):
        if isinstance(self.index, int):
            self.index = IndexValue(self.index)

# Kept-here for the example, but belong's to ARC-AGI syntax tree module
# In fact, it should become "MetadataNode", as it encapsulate data that's not in the
# main target alphabet the "program" produces'
@dataclass(frozen=True)
class RootNode(MetaNode[T]):
    """Root node: it wraps the program's node with its starting context."""
    position: CoordValue | VariableNode  # Starting position
    colors: PaletteValue | VariableNode  # Colors used in the shape

    def bit_length(self) -> int:
        pos_len = self.position.bit_length()
        colors_len = self.colors.bit_length()
        return super().bit_length() + pos_len + colors_len

    def __str__(self) -> str:
        pos_str = str(self.position)
        colors_str = str(self.colors)
        node_str = str(self.node)
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


## Functions

# Helpers
def shift_f(node: KNode[T], k: int) -> KNode[T]:
    if isinstance(node, PrimitiveNode) and isinstance(node.value, Alphabet):
        shifted_value = node.value.shift(k)
        return PrimitiveNode(shifted_value)
    return node

def next_layer(layer: Iterable[KNode]) -> tuple[KNode, ...] :
    """Used for BFS-like traversal of a K-Tree. It's basically `children` for iterable"""
    return tuple(child for node in layer for child in children(node))


# Only useful for ARC? What if the alphabet is too large?
def get_iterator(nodes: Iterable[KNode[T]]) -> tuple[KNode[T], ...]:
    """
    This function identifies if the input nodes form an arithmetic sequence and encodes it as a single RepeatNode.
    It leverages a hacky double meaning: while a standard RepeatNode(X, N) means "X for _ in range(N)",
    when used as the sole child of a SumNode, e.g., SumNode((Repeat(X, N),)), it represents
    "SumNode(tuple(shift(X, k) for k in range(N)))" if N > 0, or shifts with negative increments if N < 0.
    This compresses arithmetic enumerations cost-free, enhancing expressiveness.
    """
    node_ls: list[KNode[T]] = list(nodes)
    if len(node_ls) < 2:
        return tuple(node_ls)

    # Check if the sequence has a consistent increment (1 or -1)
    prev = node_ls[0]
    curr = node_ls[1]

    # Determine the increment based on the first pair
    if curr == shift(prev, 1):
        increment = 1
    elif curr == shift(prev, -1):
        increment = -1
    else:
        return tuple(node_ls)

    # Validate the entire sequence
    for i in range(2, len(node_ls)):
        prev = node_ls[i - 1]
        curr = node_ls[i]
        if curr != shift(prev, increment):
            return tuple(node_ls)

    # If we reach here, the sequence is arithmetic; encode it as a RepeatNode
    # Use increment * length to encode direction and length
    return (RepeatNode(node_ls[0], increment * len(node_ls)),)

# Constructors

def construct_product_node(nodes: Iterable[KNode[T]]) -> ProductNode[T]:
    """
    Constructs a ProductNode from an iterable of KNode[T], applying simplifications for efficient compression.

    - Flattens nested ProductNodes to avoid unnecessary nesting.
    - Merges adjacent PrimitiveNodes into RepeatNodes where possible using run-length encoding.
    - Combines consecutive RepeatNodes with the same base node and fixed counts.
    - Preserves SumNodes as-is to maintain logical structure.

    Args:
        nodes: An iterable of KNode[T] to combine into a ProductNode.

    Returns:
        A ProductNode[T] representing the simplified sequence.
    """
    # Convert the iterable to a list for manipulation
    simplified: list[KNode[T]] = []
    for node in nodes:
        match node:
            case ProductNode(children):
                simplified.extend(children)  # Flatten nested ProductNodes
            case _:
                simplified.append(node)  # Keep other nodes as-is

    # Simplify the flattened list
    i = 0
    while i < len(simplified):
        current = simplified[i]
        match current:
            case PrimitiveNode():
                # Collect consecutive PrimitiveNodes
                primitives = [current]
                j = i + 1
                while j < len(simplified) and isinstance(simplified[j], PrimitiveNode):
                    primitives.append(simplified[j])
                    j += 1
                if len(primitives) > 1:
                    # Replace with run-length encoded version (assumes encode_run_length exists)
                    encoded = encode_run_length(primitives)
                    simplified[i:j] = list(encoded.children)  # Insert the children directly                    i += 1
                    i += len(encoded.children)  # Move index past inserted elements
                else:
                    i += 1
            case RepeatNode(node=base, count=CountValue(count)) if i + 1 < len(simplified):
                next_node = simplified[i + 1]
                match next_node:
                    case RepeatNode(node=next_base, count=CountValue(next_count)) if base == next_base:
                        # Combine consecutive RepeatNodes with the same base
                        combined_count = CountValue(count + next_count)
                        simplified[i] = RepeatNode(base, combined_count)
                        del simplified[i + 1]
                    case _:
                        i += 1
            case _:
                i += 1  # Move past non-mergeable nodes (e.g., SumNode)

    return ProductNode(tuple(simplified))

def iterable_to_product(iterable: Iterable[KNode[T]]) -> KNode[T] | None:
    nodes: list[KNode[T]] = list(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return nodes[0]
    else:
        return construct_product_node(nodes)

def iterable_to_sum(iterable: Iterable[KNode[T]]) -> KNode[T] | None:
    nodes: list[KNode[T]] = list(iterable)
    if not nodes:
        return None
    elif len(nodes) == 1:
        return nodes[0]
    else:
        return SumNode(get_iterator(nodes))


# Traversal

def children(knode: KNode) -> Iterator[KNode]:
    """Unified API to access children  of standard KNodes nodes"""
    match knode:
        case TupleNode(children):
            return iter(children)
        case MetaNode(node):
            return iter((node,))
        case SymbolicNode(_, parameters):
            if isinstance(parameters, Variable):
                return iter((parameters,))
            return iter((param for param in parameters if isinstance(param, KNode)))
        case _:
            return iter(())

def breadth_iter(node: KNode | None) -> Iterator[KNode]:
    """
    Performs a breadth-first traversal of a KNode tree, yielding all nodes level by level.

    Args:
        node: The root KNode to traverse.

    Yields:
        KNode: Each node in the tree in breadth-first order.
    """
    if node is None:
        return
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current
        for child in children(current):
            if child is not None:
                queue.append(child)


def depth(node: KNode) -> Iterator[KNode]:
    if node is None:
        return 0

    max_depth = 0
    layer = (node,)
    while queue:
        max_depth += 1
        layer = next_layer(layer)

    return depth

# Basic operations on Nodes
def reverse_node(knode: KNode) -> KNode:
    """
    Reverses the structure of the given KNode based on its type.
    Assume that SumNode has already been run-lenght-encoded

    Args:
        knode: The KNode to reverse.

    Returns:
        KNode: The reversed node.
    """
    match knode:
        case TupleNode(children):
            reversed_children = [reverse_node(child) for child in reversed(children)]
            return type(knode)(tuple(reversed_children))
        case RepeatNode(node, count):
            return RepeatNode(reverse_node(node), count)
        case SymbolNode(index, parameters):
            reversed_params = tuple(reverse_node(p) if isinstance(p, KNode) else p for p in reversed(parameters))
            return SymbolNode(index, reversed_params)
        case RootNode(node, position, colors):
            nnode = reverse_node(node) if node else None
            return RootNode(nnode, position, colors)
        case _:
            return node

def shift(node: KNode[T], k: int) -> KNode[T]:
    """
    Shifts the values of all PrimitiveNodes in the KolmogorovTree that contain Alphabet subclasses by k.

    This function recursively traverses the KolmogorovTree using the kmap function and applies the shift
    operation to any PrimitiveNode that holds a value of a type that is a subclass of Alphabet. The shift
    operation is defined by the `shift` method of the Alphabet subclass.

    Parameters:
    -----------
    node : KNode[T]
        The root node of the KolmogorovTree to shift.
    k : int
        The amount to shift the values by.

    Returns:
    --------
    KNode[T]
        A new KolmogorovTree with the same structure, but with all shiftable values shifted by k.

    Examples:
    ---------
    >>> node = PrimitiveNode(MoveValue(1))
    >>> shifted = shift(node, 1)
    >>> shifted.value.value  # Assuming MoveValue shifts modulo 8
    2
    >>> node = ProductNode((PrimitiveNode(MoveValue(1)), PrimitiveNode(CountValue(2))))
    >>> shifted = shift(node, 1)
    >>> shifted.children[0].value.value  # Shifted MoveValue
    2
    >>> shifted.children[1].value.value  # Unchanged CountValue
    2
    """
    return kmap(node, lambda n: shift_f(n, k))

def encode_run_length(primitives: Iterable[PrimitiveNode[T]]) -> ProductNode[T]:
    """
    Encodes an iterable of PrimitiveNode[T] into a ProductNode[T] using run-length encoding.
    Consecutive identical PrimitiveNodes are compressed into RepeatNodes when the sequence
    length is 3 or greater.

    Args:
        primitives: An iterable of PrimitiveNode[T] objects.

    Returns:
        A ProductNode[T] containing a tuple of KNode[T] (PrimitiveNode[T] or RepeatNode[T]).
    """
    # List to store the sequence of nodes
    sequence: list[KNode[T]] = []

    # Convert the iterable to an iterator and handle the empty case
    iterator = iter(primitives)
    try:
        current = next(iterator)  # Get the first node
    except StopIteration:
        return ProductNode(tuple())  # Return an empty ProductNode if iterable is empty

    # Initialize count for the current sequence
    count = 1

    # Process each node in the iterable
    for node in iterator:
        if node == current:
            # If the node matches the current one, increment the count
            count += 1
        else:
            # If the node differs, decide how to encode the previous sequence
            if count >= 3:
                # For sequences of 3 or more, use a RepeatNode
                repeat_node = RepeatNode(current, CountValue(count))
                sequence.append(repeat_node)
            else:
                # For sequences of 1 or 2, append individual PrimitiveNodes
                sequence.extend([current] * count)
            # Update current node and reset count
            current = node
            count = 1

    # Handle the last sequence after the loop
    if count >= 3:
        repeat_node = RepeatNode(current, CountValue(count))
        sequence.append(repeat_node)
    else:
        sequence.extend([current] * count)

    # Return the sequence as a ProductNode
    return ProductNode(tuple(sequence))

# Tests
def is_symbolized(node: KNode) -> bool:
    """Return True if and only if node contains at least one SymbolNode in its subnodes"""
    subnodes = breadth_iter(node)
    return any(isinstance(node, SymbolNode) for node in subnodes)

# Retrievial
def contained_symbols(node: KNode) -> tuple[KNode, ...]:
    symbols = []
    layer = [node]

    while layer:
        symbols.extend([node.index for node in layer if isinstance(node, SymbolNode)])
        layer = next_layer(layer)

    return tuple(symbols)


# High-order functions
def kmap(knode: KNode, f: Callable[[KNode], KNode]) -> KNode | None:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    match knode:
        case SumNode(children):
            mapped_children = tuple(node for child in children if (node := kmap(child, f)) is not None)
            return SumNode(mapped_children)
        case ProductNode(children):
            # TODO
            # Add a compression step here
            # When the compression step is ready
            mapped_children = tuple(node for child in children if (node := kmap(child, f)) is not None)
            return ProductNode(mapped_children)
        case RepeatNode(node, count):
            nnode = kmap(node, f)
            return f(RepeatNode(nnode, count)) if nnode is not None else None
        case RootNode(node, position, color):
            nnode = kmap(node, f)
            return f(RootNode(nnode, position, color))
        case SymbolNode(index, parameters):
            nparameters = tuple(
                nparam for p in parameters
                if (nparam := (kmap(p, f) if isinstance(p, KNode) else p)) is not None
            )
            return f(SymbolNode(index, nparameters))
        case _:
            return f(knode)

# Symbol resolution and pattern finding
def replace_variables(template: KNode, params: tuple[Any, ...]) -> KNode:
    """
    Replaces variable placeholders in the template with the corresponding parameters.

    This function recursively traverses the template node and replaces VariableNodes
    with the provided parameters. It also handles composite nodes by replacing variables
    in their children or attributes.

    Args:
        template (KNode): The template node to process.
        params (tuple[Any, ...]): The parameters to substitute for variables.

    Returns:
        KNode: The template with variables replaced by parameters.
    """
    match template:
        case VariableNode(index) if index.value < len(params):
            param = params[index.value]
            return param if isinstance(param, KNode) else PrimitiveNode(param)
        case ProductNode(children):
            new_children = [replace_variables(child, params) for child in children]
            return ProductNode(new_children)
        case SumNode(children):
            new_children = [replace_variables(child, params) for child in children]
            return SumNode(new_children)
        case RepeatNode(node, count):
            new_node = replace_variables(node, params)
            new_count = replace_variables(count, params) if isinstance(count, KNode) else count
            return RepeatNode(new_node, new_count)
        case SymbolNode(index, parameters):
            new_params = tuple(
                replace_variables(p, params) if isinstance(p, KNode) else p
                for p in parameters
            )
            return SymbolNode(index, new_params)
        case RootNode(node, position, colors):
            new_pos = replace_variables(position, params) if isinstance(position, KNode) else position
            new_colors = replace_variables(colors, params) if isinstance(colors, KNode) else colors
            new_node = replace_variables(node, params) if node else None
            return RootNode(new_pos, new_colors, new_node)
        case _:
            return template

def resolve_symbols(knode: KNode, symbols: list[KNode]) -> KNode:
    """
    Resolves symbolic references recursively using the symbol table.

    This function traverses the Kolmogorov Tree and replaces SymbolNodes with their
    definitions from the symbol table, handling parameter substitution. It recursively
    resolves all sub-nodes for composite node types.

    Args:
        node (KNode): The node to resolve.
        symbols (list[KNode]): The list of symbol definitions.

    Returns:
        KNode: The resolved node with symbols expanded.
    """
    match knode:
        case SymbolNode(index, parameters) if 0 <= index.value < len(symbols):
            return replace_variables(symbols[index.value], parameters)
        case ProductNode(children):
            resolved_children = [resolve_symbols(child, symbols) for child in children]
            return ProductNode(resolved_children)
        case SumNode(children):
            resolved_children = [resolve_symbols(child, symbols) for child in children]
            return SumNode(resolved_children)
        case RepeatNode(node, count):
            resolved_node = resolve_symbols(node, symbols)
            resolved_count = resolve_symbols(count, symbols) if isinstance(count, KNode) else count
            return RepeatNode(resolved_node, resolved_count)
        case RootNode(node, position, colors):
            resolved_node = resolve_symbols(node, symbols) if node else None
            resolved_position = resolve_symbols(position, symbols) if isinstance(position, KNode) else position
            resolved_colors = resolve_symbols(colors, symbols) if isinstance(colors, KNode) else colors
            return RootNode(resolved_node, resolved_position, resolved_colors)
        case _:
            return knode

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
            return RootNode(new_node, node.position, node.colors)
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
    return PrimitiveNode(PaletteValue({color}))

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
    repeat = RepeatNode(ProductNode([move_right, move_down]), CountValue(3))
    alternative = SumNode([sequence, repeat])
    root = RootNode(alternative, CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree(root)
    print(f"Kolmogorov complexity: {tree.kolmogorov_complexity()} bits")
    return tree

def test_encode_run_length():
    # Test Case 1: Empty Input
    result = encode_run_length([])
    assert result == ProductNode(()), "Test Case 1 Failed: Empty input should return empty ProductNode"

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = encode_run_length([node])
    assert result == ProductNode((node,)), "Test Case 2 Failed: Single node should be wrapped in ProductNode"

    # Test Case 3: No Repeats
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), "Test Case 3 Failed: No repeats should remain uncompressed"

    # Test Case 4: Short Repeats
    nodes = [PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(1)),
             PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(2))]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), "Test Case 4 Failed: Repeats less than 3 should not be compressed"

    # Test Case 5: Long Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = encode_run_length(nodes)
    expected = ProductNode((RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),))
    assert result == expected, "Test Case 5 Failed: Three identical nodes should be compressed"

    # Test Case 6: Mixed Sequences
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))] + [PrimitiveNode(MoveValue(3))] * 2
    result = encode_run_length(nodes)
    expected = ProductNode((
        RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(3)),
        PrimitiveNode(MoveValue(3))
    ))
    assert result == expected, "Test Case 6 Failed: Mixed sequence compression incorrect"

    # Test Case 7: Multiple Repeats
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))] * 4
    result = encode_run_length(nodes)
    expected = ProductNode((
        RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
        RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(4))
    ))
    assert result == expected, "Test Case 7 Failed: Multiple repeat sequences not handled correctly"

    # Test Case 8: All Identical
    nodes = [PrimitiveNode(MoveValue(5))] * 10
    result = encode_run_length(nodes)
    expected = ProductNode((RepeatNode(PrimitiveNode(MoveValue(5)), CountValue(10)),))
    assert result == expected, "Test Case 8 Failed: All identical nodes should compress into one RepeatNode"

    # Test Case 9: Input as Iterator
    nodes = iter([PrimitiveNode(MoveValue(1))] * 4)
    result = encode_run_length(nodes)
    expected = ProductNode((RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(4)),))
    assert result == expected, "Test Case 9 Failed: Iterator input not handled correctly"

    # Test Case 10: Different Type
    nodes = [PrimitiveNode(PaletteValue({1}))] * 3
    result = encode_run_length(nodes)
    expected = ProductNode((RepeatNode(PrimitiveNode(PaletteValue({1})), CountValue(3)),))
    assert result == expected, "Test Case 10 Failed: Different type not processed correctly"

    # Test Case 11: Alternating Nodes
    nodes = [PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)),
             PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), "Test Case 11 Failed: Alternating nodes should not be compressed"

    # Test Case 12: Repeats in Different Positions
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))] + \
            [PrimitiveNode(MoveValue(3))] * 3 + [PrimitiveNode(MoveValue(4))] * 2
    result = encode_run_length(nodes)
    expected = ProductNode((
        RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
        PrimitiveNode(MoveValue(2)),
        RepeatNode(PrimitiveNode(MoveValue(3)), CountValue(3)),
        PrimitiveNode(MoveValue(4)),
        PrimitiveNode(MoveValue(4))
    ))
    assert result == expected, "Test Case 12 Failed: Repeats in different positions not handled correctly"

    print("Test 5: `encode_run_length` tests - Passed")

def test_construct_product_node():
    # Test 1: Empty Input
    # Verifies that an empty iterable returns an empty ProductNode
    result = construct_product_node([])
    expected = ProductNode(())
    assert result == expected, "Test 1 Failed: Empty input should return empty ProductNode"

    # Test 2: Single Node
    # Checks that a single node is wrapped in a ProductNode without changes
    node = PrimitiveNode(MoveValue(1))
    result = construct_product_node([node])
    expected = ProductNode((node,))
    assert result == expected, "Test 2 Failed: Single node should be wrapped in ProductNode"

    # Test 3: Merging Adjacent PrimitiveNodes
    # Ensures consecutive identical PrimitiveNodes are compressed into a RepeatNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3
    result = construct_product_node(nodes)
    expected = ProductNode((RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),))
    assert result == expected, "Test 3 Failed: Adjacent PrimitiveNodes should merge into RepeatNode"

    # Test 4: Preserving SumNodes
    # Confirms that SumNodes are kept intact and not merged
    sum_node = SumNode([PrimitiveNode(MoveValue(3)), PrimitiveNode(MoveValue(4))])
    nodes = [PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2))]
    result = construct_product_node(nodes)
    expected = ProductNode((PrimitiveNode(MoveValue(1)), sum_node, PrimitiveNode(MoveValue(2))))
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
    assert isinstance(shifted, PrimitiveNode), "Shifted node should be a PrimitiveNode"
    assert isinstance(shifted.value, MoveValue), "Shifted value should be a MoveValue"
    assert shifted.value.value == 3, "MoveValue should shift from 2 to 3 with k=1"

    # Test 2: Shifting by 0 (no change)
    shifted_zero = shift(node, 0)
    assert shifted_zero == node, "Shifting by 0 should return the same node"
    assert shifted_zero.value.value == 2, "Value should remain 2 when k=0"

    # Test 3: Shifting non-shiftable node
    color_node = create_color_node(1)
    shifted_color = shift(color_node, 5)
    assert shifted_color == color_node, "Non-shiftable node should remain unchanged"
    assert shifted_color.value.value == {1}, "PaletteValue should not be shifted"

    # Test 4: Shifting ProductNode with mixed primitives
    move1 = create_move_node(1)
    move2 = create_move_node(3)
    color = create_color_node(2)
    product = ProductNode([move1, color, move2])
    shifted_product = shift(product, 2)
    assert isinstance(shifted_product, ProductNode), "Shifted result should be a ProductNode"
    assert len(shifted_product.children) == 3, "ProductNode should retain 3 children"
    assert shifted_product.children[0].value.value == 3, "First MoveValue should shift to 3"
    assert shifted_product.children[1] is color, "Non-shiftable node should be unchanged"
    assert shifted_product.children[2].value.value == 5, "Second MoveValue should shift to 5"

    # Test 5: Shifting SumNode
    sum_node = SumNode([create_move_node(0), create_move_node(4)])
    shifted_sum = shift(sum_node, 2)
    assert isinstance(shifted_sum, SumNode), "Shifted result should be a SumNode"
    assert len(shifted_sum.children) == 2, "SumNode should retain 2 children"
    assert shifted_sum.children[0].value.value == 2, "First child should shift to 2"
    assert shifted_sum.children[1].value.value == 6, "Second child should shift to 6"

    # Test 6: Shifting RepeatNode
    sequence = ProductNode([create_move_node(2), create_move_node(3)])
    repeat = RepeatNode(sequence, CountValue(3))
    shifted_repeat = shift(repeat, 1)
    assert isinstance(shifted_repeat, RepeatNode), "Shifted result should be a RepeatNode"
    assert isinstance(shifted_repeat.node, ProductNode), "Repeated node should be a ProductNode"
    assert shifted_repeat.node.children[0].value.value == 3, "First MoveValue should shift to 3"
    assert shifted_repeat.node.children[1].value.value == 4, "Second MoveValue should shift to 4"
    assert shifted_repeat.count.value == 3, "Count should remain unchanged"

    # Test 7: Shifting SymbolNode
    param1 = create_move_node(1)
    param2 = create_color_node(2)
    symbol = SymbolNode(IndexValue(0), (param1, param2))
    shifted_symbol = shift(symbol, 1)
    assert isinstance(shifted_symbol, SymbolNode), "Shifted result should be a SymbolNode"
    assert len(shifted_symbol.parameters) == 2, "SymbolNode should retain 2 parameters"
    assert shifted_symbol.parameters[0].value.value == 2, "MoveValue parameter should shift to 2"
    assert shifted_symbol.parameters[1] is param2, "Non-shiftable parameter should be unchanged"

    # Test 8: Shifting RootNode
    program = ProductNode([create_move_node(0), create_move_node(1)])
    root = RootNode(program, CoordValue((0, 0)), PaletteValue({1}))
    shifted_root = shift(root, 2)
    assert isinstance(shifted_root, RootNode), "Shifted result should be a RootNode"
    assert isinstance(shifted_root.node, ProductNode), "Root program should be a ProductNode"
    assert shifted_root.node.children[0].value.value == 2, "First MoveValue should shift to 2"
    assert shifted_root.node.children[1].value.value == 3, "Second MoveValue should shift to 3"
    assert shifted_root.position == CoordValue((0, 0)), "Position should be unchanged"
    assert shifted_root.colors == PaletteValue({1}), "Colors should be unchanged"

    # Test 9: Shifting with large k (wrapping around)
    node_large = create_move_node(7)
    shifted_large = shift(node_large, 10)  # 7 + 10 = 17 ≡ 1 mod 8
    assert shifted_large.value.value == 1, "Large shift should wrap around to 1"

    # Test 10: Shifting with negative k
    shifted_neg = shift(node_large, -3)  # 7 - 3 = 4
    assert shifted_neg.value.value == 4, "Negative shift should result in 4"

    # Test 11: Shifting nested composite nodes
    inner_product = ProductNode([create_move_node(5), create_move_node(6)])
    repeat_inner = RepeatNode(inner_product, CountValue(2))
    outer_sum = SumNode([repeat_inner, create_move_node(7)])
    shifted_outer = shift(outer_sum, 1)
    assert isinstance(shifted_outer, SumNode), "Shifted outer node should be a SumNode"
    assert isinstance(shifted_outer.children[0], RepeatNode), "First child should be a RepeatNode"
    assert isinstance(shifted_outer.children[0].node, ProductNode), "Repeated node should be ProductNode"
    assert shifted_outer.children[0].node.children[0].value.value == 6, "First nested MoveValue should shift to 6"
    assert shifted_outer.children[0].node.children[1].value.value == 7, "Second nested MoveValue should shift to 7"
    assert shifted_outer.children[1].value.value == 0, "Outer MoveValue should shift from 7 to 0"

    # Test 12: Original node remains unchanged
    original_node = create_move_node(2)
    shifted_node = shift(original_node, 1)
    assert original_node.value.value == 2, "Original node value should remain 2"
    assert shifted_node.value.value == 3, "Shifted node value should be 3"

    print("Test shift operations - Passed")

def test_get_iterator():
    """Tests the get_iterator function for detecting and compressing arithmetic sequences."""
    # Test Case 1: Empty Input
    result = get_iterator([])
    assert result == (), "Test Case 1 Failed: Empty input should return empty tuple"

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = get_iterator([node])
    assert result == (node,), "Test Case 2 Failed: Single node should return tuple with that node"

    # Test Case 3: Sequence with Increment +1
    nodes_pos = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]]
    result_pos = get_iterator(nodes_pos)
    expected_pos = (RepeatNode(nodes_pos[0], 3),)
    assert result_pos == expected_pos, "Test Case 3 Failed: Sequence [0,1,2] should be compressed to RepeatNode(0, 3)"

    # Test Case 4: Sequence with Increment -1
    nodes_neg = [PrimitiveNode(MoveValue(i)) for i in [2, 1, 0]]
    result_neg = get_iterator(nodes_neg)
    expected_neg = (RepeatNode(nodes_neg[0], CountValue(-3)),)
    assert result_neg == expected_neg, "Test Case 4 Failed: Sequence [2,1,0] should be compressed to RepeatNode(2, -3)"

    # Test Case 5: Non-Sequence
    nodes_non = [PrimitiveNode(MoveValue(i)) for i in [0, 2, 1]]
    result_non = get_iterator(nodes_non)
    assert result_non == tuple(nodes_non), "Test Case 5 Failed: Non-sequence should return original nodes"

    # Test Case 6: Boundary Conditions (Wrap-around with Increment +1)
    nodes_wrap = [PrimitiveNode(MoveValue(i)) for i in [7, 0, 1]]
    result_wrap = get_iterator(nodes_wrap)
    expected_wrap = (RepeatNode(nodes_wrap[0], 3),)
    assert result_wrap == expected_wrap, "Test Case 7 Failed: Wrap-around sequence [7,0,1] should be compressed"

    # Test Case 7: Long Sequence with Wrap-around
    nodes_long = [PrimitiveNode(MoveValue(i % 8)) for i in range(10)]  # [0,1,2,3,4,5,6,7,0,1]
    result_long = get_iterator(nodes_long)
    expected_long = (RepeatNode(nodes_long[0], CountValue(10)),)
    assert result_long == expected_long, "Test Case 8 Failed: Long sequence [0,1,2,3,4,5,6,7,0,1] should be compressed"

    # Test Case 8: Partial Sequence
    nodes_partial = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2, 4]]
    result_partial = get_iterator(nodes_partial)
    assert result_partial == tuple(nodes_partial), "Test Case 9 Failed: Partial sequence [0,1,2,4] should not be compressed"

    # Test Case 9: Different Increment
    nodes_diff_inc = [PrimitiveNode(MoveValue(i)) for i in [0, 2, 4, 6]]
    result_diff_inc = get_iterator(nodes_diff_inc)
    assert result_diff_inc == tuple(nodes_diff_inc), "Test Case 10 Failed: Sequence with increment +2 should not be compressed"

    # Test Case 10: Changing Increment
    nodes_change_inc = [PrimitiveNode(MoveValue(i)) for i in [0, 1, 2, 1, 0]]
    result_change_inc = get_iterator(nodes_change_inc)
    assert result_change_inc == tuple(nodes_change_inc), "Test Case 11 Failed: Sequence with changing increment should not be compressed"

    print("Test get_iterator - Passed")

def run_tests():
    """Runs simple tests to verify KolmogorovTree functionality."""
    # Test 1: Basic node creation and bit length
    move_right = PrimitiveNode(MoveValue(2))
    assert move_right.bit_length() == 6, "PrimitiveNode bit length should be 6 (3 + 3)"
    product = ProductNode([move_right, move_right])
    assert product.bit_length() == 15, "ProductNode bit length should be 15 (3 + 6 + 6)"
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode([move_right, move_down])
    assert sum_node.bit_length() == 17, "SumNode bit length should be 17 (3 + 2 + 6 + 6)"
    repeat = RepeatNode(move_right, CountValue(3))
    assert repeat.bit_length() == 14, "RepeatNode bit length should be 14 (3 + 6 + 5)"
    symbol = SymbolNode(IndexValue(0), ())
    assert symbol.bit_length() == 10, "SymbolNode bit length should be 10 (3 + 7)"
    root = RootNode(product, CoordValue((0, 0)), PaletteValue({1}))
    assert root.bit_length() == 32, "RootNode bit length should be 32 (3 + 15 + 10 + 4)"
    print("Test 1: Basic node creation and bit length - Passed")

    # Test 2: String representations
    assert str(move_right) == "2", "PrimitiveNode str should be '2'"
    assert str(product) == "22", "ProductNode str should be '22'"
    assert str(sum_node) == "[2|3]", "SumNode str should be '[2|3]'"
    assert str(repeat) == "(2)*{3}", "RepeatNode str should be '(2)*{3}'"
    assert str(symbol) == "s_0", "SymbolNode str should be 's_0'"
    print("root: ", root)
    assert str(root) == "Root((0, 0), {1}, 22)", "RootNode str should be 'Root((0, 0), {1}, 22)'"
    print("Test 2: String representations - Passed")

    # Test 3: Operator overloads
    assert (move_right | move_down) == SumNode([move_right, move_down]), "Operator | failed"
    assert (move_right & move_down) == ProductNode([move_right, move_down]), "Operator & failed"
    assert (move_right + move_down) == ProductNode([move_right, move_down]), "Operator + failed"
    assert (move_right * 3) == RepeatNode(move_right, CountValue(3)), "Operator * failed"
    assert ((move_right * 2) * 3) == RepeatNode(move_right, CountValue(6)), "Nested * failed"
    print("Test 3: Operator overloads - Passed")

    # Test 4: Symbol resolution
    symbol_def = ProductNode([move_right, move_down])
    symbols = [symbol_def]
    symbol_node = SymbolNode(IndexValue(0), ())
    root_with_symbol = RootNode(ProductNode([symbol_node, move_right]), CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree(root_with_symbol, symbols)
    resolved = resolve_symbols(tree.root, tree.symbols)
    expected_resolved = RootNode(ProductNode([symbol_def, move_right]), CoordValue((0, 0)), PaletteValue({1}))
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

    test_encode_run_length()
    test_construct_product_node()
    test_shift()
    test_get_iterator()

if __name__ == "__main__":
    tree = example_usage()
    run_tests()
    print("All tests passed successfully!")
