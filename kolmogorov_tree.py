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


from typing import Generic, TypeVar, Callable, Iterable, Any, Iterator, Self, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import Counter, deque
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
    def shift(self, k: int) -> Self:
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

    def shift(self, k: int) -> 'MoveValue':
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
class KNode(Generic[T], BitLengthAware, ABC):
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

    @property
    def data(self) -> Any:
        """Returns the underlying data of the PrimitiveNode."""
        return self.value.value

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
        # Horrible hack for the already horrible iterator hack
        if len(self.children) == 1 and isinstance(self.children[0], RepeatNode):
            return "[+" + str(self.children[0]) + "]"
        return "[" + "|".join(str(child) for child in self.children) + "]"

@dataclass(frozen=True)
class MetaNode(KNode[T], ABC):
    """Wraps a single node with additional information. node allows None to handle edge cases for map functions"""
    node: KNode[T]

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
    reference_length: int = 0

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().bit_length() + self.index.bit_length() + params_len

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}"

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
class KolmogorovTree(Generic[T]):
    """Represents a complete program with a root node and symbol table."""
    root: KNode[T]
    symbols: list[KNode[T]] = field(default_factory=list)

    def bit_length(self) -> int:
        """Total bit length includes root and symbol definitions."""
        return self.root.bit_length() + sum(symbol.bit_length() for symbol in self.symbols)

    def resolve_symbols(self) -> KNode[T]:
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
        return PrimitiveNode[T](shifted_value)
    return node

def next_layer(layer: Iterable[KNode]) -> tuple[KNode, ...] :
    """Used for BFS-like traversal of a K-Tree. It's basically `children` for iterable"""
    return tuple(child for node in layer for child in children(node))

Parameters = tuple[BitLengthAware, ...]

def generate_abstraction(knode: KNode[T]) -> list[tuple[KNode[T], Parameters]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions: list[tuple[KNode[T], Parameters] = []

    match knode:
        case TupleNode(children) if len(children) > 2:
            pass
        case RepeatNode(node, count):
            pass
        case Root(node, position, colors):
            pass
        #case Rect()
            #pass

    return abstractions

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
    return (RepeatNode(node_ls[0], CountValue(increment * len(node_ls))),)

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
                    primitives.append(simplified[j]) # type: ignore[reportArgumentType]
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
    """Unified API to access children of standard KNodes nodes"""
    match knode:
        case TupleNode(children):
            return iter(children)
        case MetaNode(node):
            return iter((node,))
        case SymbolNode(_, parameters):
            if isinstance(parameters, VariableNode):
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
        return iter(())
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current
        for child in children(current):
            if child is not None:
                queue.append(child)


def depth(node: KNode) -> int:
    """Returns the depth of a Kolmogorov tree"""
    max_depth = 0
    layer = (node,)
    while layer:
        max_depth += 1
        layer = next_layer(layer)

    return max_depth

# Basic operations on Nodes
def reverse_node(knode: KNode[T]) -> KNode[T]:
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
            nnode = reverse_node(node)
            return RootNode(nnode, position, colors)
        case _:
            return knode

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
    >>> shifted.data  # Assuming MoveValue shifts modulo 8
    2
    >>> node = ProductNode((PrimitiveNode(MoveValue(1)), PrimitiveNode(CountValue(2))))
    >>> shifted = shift(node, 1)
    >>> shifted.children[0].data  # Shifted MoveValue
    2
    >>> shifted.children[1].data  # Unchanged CountValue
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

def is_function(node: KNode) -> bool:
    """Return True if and only if node contains at least one VariableNode in its subnodes"""
    subnodes = breadth_iter(node)
    return any(isinstance(node, VariableNode) for node in subnodes)

# Retrievial
def contained_symbols(knode: KNode) -> tuple[IndexValue, ...]:
    subnodes = breadth_iter(knode)
    return tuple(node.index for node in subnodes if isinstance(node, SymbolNode))

# Compression
def find_repeating_pattern(nodes: Sequence[KNode[T]], offset: int) -> tuple[KNode[T] | None, int, bool]:
    """
    Finds the best repeating pattern in a list of KNodes starting at the given offset, including alternating (reversed) patterns,
    optimizing for bit-length compression.

    Args:
        nodes: List of KNode[T] to search for patterns.
        offset: Starting index in the list to begin the search.

    Returns:
        Tuple of (pattern_node, count, is_reversed):
        - pattern_node: The KNode representing the repeating unit (None if no pattern found).
        - count: Number of repetitions (positive or negative based on reversal).
        - is_reversed: True if the pattern alternates with its reverse.

    """
    if offset >= len(nodes):
        return None, 0, False

    best_pattern: KNode[T] | None = None
    best_count = 0
    best_bit_gain = 0
    best_reverse = False
    max_pattern_len = (len(nodes) - offset + 1) // 2  # Need at least 2 occurrences

    for pattern_len in range(1, max_pattern_len + 1):
        pattern = nodes[offset:offset + pattern_len]
        pattern_node = construct_product_node(pattern) if len(pattern) > 1 else pattern[0]

        for reverse in [False, True]:
            count = 1
            i = offset + pattern_len
            while i < len(nodes):
                if i + pattern_len > len(nodes):
                    break

                match = True
                segment = nodes[i:i + pattern_len]
                compare_node = reverse_node(pattern_node) if (reverse and count % 2 == 1) else pattern_node

                # Compare the segment with the pattern or its reverse
                if len(segment) == pattern_len:
                    segment_node = construct_product_node(segment) if len(segment) > 1 else segment[0]
                    if segment_node != compare_node:
                        match = False
                else:
                    match = False

                if match:
                    count += 1
                    i += pattern_len
                else:
                    break

            if count > 1:
                # Calculate bit-length savings
                original_bits = sum(node.bit_length() for node in nodes[offset:offset + pattern_len * count])
                compressed = RepeatNode(pattern_node, CountValue(count if not reverse else -count))
                compressed_bits = compressed.bit_length()
                bit_gain = original_bits - compressed_bits

                if bit_gain > best_bit_gain:
                    best_pattern = pattern_node
                    best_count = count
                    best_bit_gain = bit_gain
                    best_reverse = reverse

    return best_pattern, best_count if not best_reverse else -best_count, best_reverse

def factorize_tuple(node: KNode[T]) -> KNode[T]:
    """
    Compresses a ProductNode or SumNode by detecting and encoding repeating patterns with RepeatNodes.

    Args:
        node: A KNode[T], typically a ProductNode or SumNode, to factorize.

    Returns:
        A new KNode[T] with repeating patterns compressed.
    """
    if isinstance(node, SumNode):
        # For SumNode, delegate to get_iterator for arithmetic sequence compression
        children = get_iterator(node.children)
        #return SumNode(children) if len(children) > 1 else children[0] if children else node
        # For the iterator hack, no unpacking
        return SumNode(children)

    if not isinstance(node, ProductNode):
        return node

    # Handle ProductNode compression
    children = list(node.children)
    if len(children) < 2:
        return node

    simplified: list[KNode[T]] = []
    i = 0
    while i < len(children):
        pattern, count, is_reversed = find_repeating_pattern(children, i)
        if pattern is not None and abs(count) > 1:
            repeat_node = RepeatNode(pattern, CountValue(count))
            simplified.append(repeat_node)
            i += abs(count) * (len(pattern.children) if isinstance(pattern, ProductNode) else 1)
        else:
            simplified.append(children[i])
            i += 1

    # Reconstruct the ProductNode with simplified children
    if len(simplified) == 1 and simplified[0] == node:
        return node  # Avoid unnecessary reconstruction
    return construct_product_node(simplified)

# Optional: Apply factorization across the entire tree
def factorize_tree(tree: KolmogorovTree[T]) -> KolmogorovTree[T]:
    """
    Applies factorization to all ProductNodes and SumNodes in a KolmogorovTree.

    Args:
        tree: The KolmogorovTree to factorize.

    Returns:
        A new KolmogorovTree with compressed patterns.
    """
    def factorize_node(n: KNode[T]) -> KNode[T]:
        return factorize_tuple(n)

    new_root = kmap(tree.root, factorize_node)
    new_symbols = [kmap(symbol, factorize_node) for symbol in tree.symbols]
    return KolmogorovTree(new_root, new_symbols)

# High-order functions
def kmap(knode: KNode, f: Callable[[KNode], KNode]) -> KNode:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    match knode:
        case TupleNode(children):
            mapped_children = tuple(kmap(child, f) for child in children)
            return f(factorize_tuple(type(knode)(mapped_children)))
        case RepeatNode(node, count):
            nnode = kmap(node, f)
            return f(RepeatNode(nnode, count))
        case RootNode(node, position, color):
            nnode = kmap(node, f)
            return f(RootNode(nnode, position, color))
        case SymbolNode(index, parameters):
            nparameters = tuple(
                (kmap(p, f) if isinstance(p, KNode) else p) for p in parameters
            )
            return f(SymbolNode(index, nparameters))
        case _:
            return f(knode)

def kmap_unsafe(knode: KNode, f: Callable[[KNode], KNode | None]) -> KNode | None:
    """
    Map a function alongside a KNode. It updates first childrens, then updates the base node

    Args:
        knode: The KNode tree to transform
        f: A function that takes a KNode and returns a transformed KNode, or returning None.

    Returns:
        KNode: A new KNode tree with the function f applied to each node
    """
    match knode:
        case TupleNode(children):
            mapped_children = tuple(node for child in children if (node := kmap_unsafe(child, f)) is not None)
            return f(factorize_tuple(type(knode)(mapped_children)))
        case RepeatNode(node, count):
            nnode = kmap_unsafe(node, f)
            return f(RepeatNode(nnode, count)) if nnode is not None else None
        case RootNode(node, position, color):
            nnode = kmap_unsafe(node, f)
            return f(RootNode(nnode, position, color)) if nnode is not None else None
        case SymbolNode(index, parameters):
            nparameters = tuple(
                nparam for p in parameters
                if (nparam := (kmap_unsafe(p, f) if isinstance(p, KNode) else p)) is not None
            )
            return f(SymbolNode(index, nparameters))
        case _:
            return f(knode)
# Symbol resolution and pattern finding

def generate_abstractions(node: KNode[T]) -> list[tuple[KNode[T], tuple[BitLengthAware, ...]]]:
    """Generate abstracted versions of a KNode by replacing subparts with variables."""
    abstractions = []

    match node:
        case RootNode(nnode, position, colors):
            # Avoid re-abstracting if already a variable
            # Only abstract if components are not already variables
            if (
                not isinstance(position, VariableNode)
                and not isinstance(colors, VariableNode)
                and not isinstance(nnode, VariableNode)
            ):
                # Abstract position only
                abs_pos = RootNode(
                    position=VariableNode(VariableValue(0)),
                    colors=colors,
                    node=nnode
                )
                abstractions.append((abs_pos, (position,)))

                # Abstract position and node
                abs_pos_node = RootNode(
                    position=VariableNode(VariableValue(0)),
                    colors=colors,
                    node=VariableNode(VariableValue(1))
                )
                abstractions.append((abs_pos_node, (position, nnode)))

                # If colors is a single color, abstract position and colors
                if isinstance(colors, PaletteValue) and len(colors.value) == 1:
                    abs_pos_colors = RootNode(
                        position=VariableNode(VariableValue(0)),
                        colors=VariableNode(VariableValue(1)),
                        node=nnode
                    )
                    abstractions.append((abs_pos_colors, (position, colors)))
        case ProductNode(children):
            if not any(isinstance(c, VariableNode) for c in children):
                # Sort children by bit length to prioritize replacing larger subtrees
                child_set = set(children)
                max_children = sorted(child_set, key=lambda x: x.bit_length(), reverse=True)[:2]

                # Abstract the most frequent/largest child
                nodes1 = tuple(VariableNode(VariableValue(0)) if c == max_children[0] else c for c in children)
                abstractions.append((ProductNode(nodes1), (max_children[0],)))

                # If there are at least two distinct children and length > 2, abstract the top two
                if len(max_children) > 1 and len(children) > 2:
                    nodes2 = tuple(VariableNode(VariableValue(max_children.index(c))) if c in max_children else c
                                 for c in children)
                    abstractions.append((ProductNode(nodes2), tuple(max_children)))

        case RepeatNode(nnode, count) if not isinstance(nnode, VariableNode) and not isinstance(count, VariableNode):
            # Abstract the repeated node
            abstractions.append((RepeatNode(VariableNode(VariableValue(0)), count), (nnode,)))
            # Abstract the count (if count is a CountValue, wrap it appropriately)
            if isinstance(count, BitLengthAware):
                abstractions.append((RepeatNode(nnode, VariableNode(VariableValue(0))), (count,)))

    return abstractions

def matches(pattern: KNode[T], subtree: KNode[T]) -> [dict[int, KNode[T]] | None:
    """Match a pattern with variables against a concrete subtree, returning variable bindings."""
    bindings: dict[int, KNode[T]] = {}

    def unify(p: KNode[T], s: KNode[T]) -> dict[int, KNode[T]] | None:
        if isinstance(p, VariableNode):
            idx = p.index.value
            if idx in bindings:
                return bindings if bindings[idx] == s else None
            bindings[idx] = s
            return bindings

        if type(p) != type(s):
            return None

        match p:
            case PrimitiveNode(value=pv):
                if isinstance(s, PrimitiveNode) and pv == s.value:
                    return bindings
                return None
            case RootNode(position=pp, colors=pc, node=pn):
                if isinstance(s, RootNode):
                    return (unify(pp, s.position) and
                            unify(pc, s.colors) and
                            unify(pn, s.node))
            case ProductNode(children=pc) if isinstance(s, ProductNode):
                if len(pc) != len(s.children):
                    return None
                for pc_child, sc_child in zip(pc, s.children):
                    result = unify(pc_child, sc_child)
                    if result is None:
                        return None
                return bindings
            case RepeatNode(node=pn, count=pc) if isinstance(s, RepeatNode):
                if pc != s.count:
                    return None
                return unify(pn, s.node)
            case _:
                return bindings if p == s else None

    return unify(pattern, subtree)

def replace_variables(template: KNode[T], params: tuple[Any, ...]) -> KNode[T]:
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
            new_children = tuple(replace_variables(child, params) for child in children)
            return ProductNode(new_children)
        case SumNode(children):
            new_children = tuple(replace_variables(child, params) for child in children)
            return SumNode(new_children)
        case RepeatNode(node, count):
            new_node = replace_variables(node, params)
            new_count = replace_variables(count, params) if isinstance(count, KNode) else count
            if isinstance(new_count, KNode) and not isinstance(new_count, VariableNode): # To make Pyright happy
                raise TypeError(f"New count must be CountValue or VariableNode, got {type(new_count)}")
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
            new_node = replace_variables(node, params)

            # To make Pyright happy
            if isinstance(new_pos, KNode) and not isinstance(new_pos, VariableNode):
                raise TypeError(f"New position must be CoordValue or VariableNode, got {type(new_pos)}")
            if isinstance(new_colors, KNode) and not isinstance(new_colors, VariableNode):
                raise TypeError(f"New colors must be PaletteValue or VariableNode, got {type(new_colors)}")

            return RootNode(new_node, new_pos, new_colors)
        case _:
            return template

def resolve_symbols(knode: KNode[T], symbols: list[KNode[T]]) -> KNode[T]:
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
            return ProductNode(tuple(resolved_children))
        case SumNode(children):
            resolved_children = [resolve_symbols(child, symbols) for child in children]
            return SumNode(tuple(resolved_children))
        case RepeatNode(node, count):
            resolved_node = resolve_symbols(node, symbols)
            resolved_count = resolve_symbols(count, symbols) if isinstance(count, KNode) else count
            if isinstance(resolved_count, KNode) and not isinstance(resolved_count, VariableNode): # To make Pyright happy
                raise TypeError(f"Resolved count must be CountValue or VariableNode, got {type(resolved_count)}")
            return RepeatNode(resolved_node, resolved_count)
        case RootNode(node, position, colors):
            resolved_node = resolve_symbols(node, symbols)
            resolved_position = resolve_symbols(position, symbols) if isinstance(position, KNode) else position
            resolved_colors = resolve_symbols(colors, symbols) if isinstance(colors, KNode) else colors

            # To make Pyright happy
            if isinstance(resolved_position, KNode) and not isinstance(resolved_position, VariableNode):
                raise TypeError(f"Resolved position must be CoordValue or VariableNode, got {type(resolved_position)}")
            if isinstance(resolved_colors, KNode) and not isinstance(resolved_colors, VariableNode):
                raise TypeError(f"Resolved colors must be PaletteValue or VariableNode, got {type(resolved_colors)}")

            return RootNode(resolved_node, resolved_position, resolved_colors)
        case _:
            return knode

def find_common_subtrees_exact(trees: list[KolmogorovTree], min_occurrences=2, max_patterns=10) -> list[KNode]:
    """Identifies frequent subtrees across multiple trees for symbolization using breadth_iter."""
    all_subtrees = []
    for tree in trees:
        # Collect all subtrees starting from the root
        all_subtrees.extend(breadth_iter(tree.root))
        for symbol in tree.symbols:
            all_subtrees.extend(breadth_iter(symbol))

    # Count frequencies using string representations
    counter = Counter(str(node) for node in all_subtrees)
    common_nodes = []
    seen = set()
    for node in all_subtrees:
        node_str = str(node)
        if counter[node_str] >= min_occurrences and node_str not in seen:
            common_nodes.append(node)
            seen.add(node_str)

    # Sort by frequency (descending) and bit length (descending)
    common_nodes.sort(key=lambda n: (-counter[str(n)], -n.bit_length()))
    return common_nodes[:max_patterns]

def find_common_subtrees(trees: list[KolmogorovTree], min_occurrences: int = 2, max_patterns: int = 10) -> list[KNode]:
    """Identify frequent concrete and abstracted subtrees across multiple KolmogorovTrees."""
    all_subtrees = []
    for tree in trees:
        all_subtrees.extend(breadth_iter(tree.root))
        for symbol in tree.symbols:
            all_subtrees.extend(breadth_iter(symbol))

    # Collect candidate patterns: concrete and abstracted
    pattern_counter = Counter()
    pattern_instances = defaultdict(list)  # Maps pattern to list of (subtree, params) matches

    for subtree in all_subtrees:
        # Add concrete subtree
        pattern_counter[subtree] += 1
        pattern_instances[subtree].append((subtree, ()))

        # Add abstracted versions
        for abs_pattern, params in generate_abstractions(subtree):
            pattern_counter[abs_pattern] += 1
            pattern_instances[abs_pattern].append((subtree, params))

    # Filter patterns by minimum occurrences
    common_patterns = []
    seen_patterns = set()

    for pattern, count in pattern_counter.items():
        if count >= min_occurrences and pattern not in seen_patterns:
            # For abstracted patterns, verify distinct matches
            if any(isinstance(n, VariableNode) for n in breadth_iter(pattern)):
                distinct_matches = set()
                for s, _ in pattern_instances[pattern]:
                    if str(s) not in distinct_matches:
                        distinct_matches.add(str(s))
                if len(distinct_matches) >= min_occurrences:
                    common_patterns.append(pattern)
                    seen_patterns.add(pattern)
            else:
                common_patterns.append(pattern)
                seen_patterns.add(pattern)

    # Sort by bit gain (frequency * savings) and limit to max_patterns
    def bit_gain(pat: KNode) -> int:
        count = len(set(s for s, _ in pattern_instances[pat]))  # Distinct subtrees matched
        avg_len = sum(s.bit_length() for s, _ in pattern_instances[pat]) / count
        param_len = sum(p.bit_length() for _, ps in pattern_instances[pat] for p in ps) / count
        symb_len = BitLength.NODE_TYPE + BitLength.INDEX + int(param_len)
        return (count - 1) * (avg_len - symb_len) - pat.bit_length()

    common_patterns.sort(key=lambda p: (-bit_gain(p), -p.bit_length()))
    return common_patterns[:max_patterns]

def symbolize(trees: list[KolmogorovTree]) -> Optional[KolmogorovTree]:
    """Symbolize trees by replacing common patterns (concrete or abstracted) with SymbolNodes."""
    common_patterns = find_common_subtrees(trees)
    if not common_patterns or not trees:
        return None if not trees else KolmogorovTree(trees[0].root, [])

    symbols = list(common_patterns)

    def replace_node(node: KNode[T]) -> KNode[T]:
        for i, pattern in enumerate(common_patterns):
            if pattern == node:
                return SymbolNode(IndexValue(i), (), pattern.bit_length())
            bindings = match(pattern, node)
            if bindings is not None:
                params = tuple(bindings.get(j, PrimitiveNode(0)) for j in range(max(bindings.keys(), default=-1) + 1))
                return SymbolNode(IndexValue(i), params, pattern.bit_length())

        match node:
            case ProductNode(children=ch):
                return ProductNode(tuple(replace_node(c) for c in ch))
            case SumNode(children=ch):
                return SumNode(tuple(replace_node(c) for c in ch))
            case RepeatNode(node=n, count=c):
                return RepeatNode(replace_node(n), c)
            case RootNode(node=n, position=p, colors=col):
                return RootNode(replace_node(n), p, col)
            case _:
                return node

    new_root = replace_node(trees[0].root)
    return KolmogorovTree(new_root, symbols)

# Factory functions for common patterns
def create_move_node(direction: int) -> PrimitiveNode:
    """Creates a node for a directional move."""
    return PrimitiveNode(MoveValue(direction))

def create_moves_sequence(directions: str) -> KNode | None:
    """Creates a sequence of moves from a direction string."""
    if not directions:
        return None
    moves = [create_move_node(int(d)) for d in directions]
    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len
                if all(moves[i * pattern_len:(i + 1) * pattern_len] == pattern
                       for i in range(repeat_count)) and repeat_count > 1:
                    return RepeatNode(ProductNode(tuple(pattern)), CountValue(repeat_count))
    return ProductNode(tuple(moves))

def create_rect(height: int, width: int) -> KNode | None:
    """Creates a node for a rectangle shape."""
    if height < 2 or width < 2:
        return None
    first_row = "2" * (width - 1)  # Move right
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"  # Alternate left/right
        other_rows += "3" + direction * (width - 1)  # Down then horizontal
    return create_moves_sequence(first_row + other_rows)

# Example usage
def example_usage():
    """Demonstrates creating and evaluating a KolmogorovTree."""
    move_right = create_move_node(2)
    move_down = create_move_node(3)
    sequence = ProductNode((move_right, move_right, move_down))
    repeat = RepeatNode(ProductNode((move_right, move_down)), CountValue(3))
    alternative = SumNode((sequence, repeat))
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

    # Test Case 10: Alternating Nodes
    nodes = [PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2)),
             PrimitiveNode(MoveValue(1)), PrimitiveNode(MoveValue(2))]
    result = encode_run_length(nodes)
    assert result == ProductNode(tuple(nodes)), "Test Case 11 Failed: Alternating nodes should not be compressed"

    # Test Case 11: Repeats in Different Positions
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
    sum_node = SumNode(tuple([PrimitiveNode(MoveValue(3)), PrimitiveNode(MoveValue(4))]))
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
    assert shifted.data == 3, "MoveValue should shift from 2 to 3 with k=1"

    # Test 2: Shifting by 0 (no change)
    shifted_zero = shift(node, 0)
    assert shifted_zero == node, "Shifting by 0 should return the same node"
    assert isinstance(shifted_zero, PrimitiveNode), "Shifted node should be a PrimitiveNode"
    assert shifted_zero.data == 2, "Value should remain 2 when k=0"

    # Test 3: Shifting SumNode
    sum_node = SumNode((create_move_node(0), create_move_node(4)))
    shifted_sum = shift(sum_node, 2)
    assert isinstance(shifted_sum, SumNode), "Shifted result should be a SumNode"
    assert len(shifted_sum.children) == 2, "SumNode should retain 2 children"
    assert isinstance(shifted_sum.children[0], PrimitiveNode), "First child should be PrimitiveNode"
    assert shifted_sum.children[0].data == 2, "First child should shift to 2"
    assert isinstance(shifted_sum.children[1], PrimitiveNode), "Second child should be PrimitiveNode"
    assert shifted_sum.children[1].data == 6, "Second child should shift to 6"

    # Test 4: Shifting RepeatNode
    sequence = ProductNode((create_move_node(2), create_move_node(3)))
    repeat = RepeatNode(sequence, CountValue(3))
    shifted_repeat = shift(repeat, 1)
    assert isinstance(shifted_repeat, RepeatNode), "Shifted result should be a RepeatNode"
    assert isinstance(shifted_repeat.node, ProductNode), "Repeated node should be a ProductNode"
    assert isinstance(shifted_repeat.node.children[0], PrimitiveNode), "First child should be PrimitiveNode"
    assert shifted_repeat.node.children[0].data == 3, "First MoveValue should shift to 3"
    assert isinstance(shifted_repeat.node.children[1], PrimitiveNode), "Second child should be PrimitiveNode"
    assert shifted_repeat.node.children[1].data == 4, "Second MoveValue should shift to 4"
    assert isinstance(shifted_repeat.count, CountValue), "COunt should be CountValue"
    assert shifted_repeat.count.value == 3, "Count should remain unchanged"

    # Test 5: Shifting SymbolNode
    param1 = create_move_node(1)
    param2 = PaletteValue({2})
    symbol = SymbolNode(IndexValue(0), (param1, param2))
    shifted_symbol = shift(symbol, 1)
    assert isinstance(shifted_symbol, SymbolNode), "Shifted result should be a SymbolNode"
    assert len(shifted_symbol.parameters) == 2, "SymbolNode should retain 2 parameters"
    assert isinstance(shifted_symbol.parameters[0], PrimitiveNode), "Parameter should be PrimitiveNode"
    assert shifted_symbol.parameters[0].data == 2, "MoveValue parameter should shift to 2"
    assert shifted_symbol.parameters[1] is param2, "Non-shiftable parameter should be unchanged"

    # Test 6: Shifting RootNode
    program = ProductNode((create_move_node(0), create_move_node(1)))
    root = RootNode(program, CoordValue((0, 0)), PaletteValue({1}))
    shifted_root = shift(root, 2)
    assert isinstance(shifted_root, RootNode), "Shifted result should be a RootNode"
    assert isinstance(shifted_root.node, ProductNode), "Root program should be a ProductNode"
    assert isinstance(shifted_root.node.children[0], PrimitiveNode), "First child should be PrimitiveNode"
    assert shifted_root.node.children[0].data == 2, "First MoveValue should shift to 2"
    assert isinstance(shifted_root.node.children[1], PrimitiveNode), "Second child should be PrimitiveNode"
    assert shifted_root.node.children[1].data == 3, "Second MoveValue should shift to 3"
    assert shifted_root.position == CoordValue((0, 0)), "Position should be unchanged"
    assert shifted_root.colors == PaletteValue({1}), "Colors should be unchanged"

    # Test 7: Shifting with large k (wrapping around)
    node_large = create_move_node(7)
    shifted_large = shift(node_large, 10)  # 7 + 10 = 17 ≡ 1 mod 8
    assert isinstance(shifted_large, PrimitiveNode), "Shifted node should be a PrimitiveNode"
    assert shifted_large.data == 1, "Large shift should wrap around to 1"

    # Test 8: Shifting with negative k
    shifted_neg = shift(node_large, -3)  # 7 - 3 = 4
    assert isinstance(shifted_neg, PrimitiveNode), "Shifted node should be a PrimitiveNode"
    assert shifted_neg.data == 4, "Negative shift should result in 4"

    # Test 9: Shifting nested composite nodes
    inner_product = ProductNode((create_move_node(5), create_move_node(6)))
    repeat_inner = RepeatNode(inner_product, CountValue(2))
    outer_sum = SumNode((repeat_inner, create_move_node(7)))
    shifted_outer = shift(outer_sum, 1)
    assert isinstance(shifted_outer, SumNode), "Shifted outer node should be a SumNode"
    assert isinstance(shifted_outer.children[0], RepeatNode), "First child should be a RepeatNode"
    assert isinstance(shifted_outer.children[0].node, ProductNode), "Repeated node should be ProductNode"
    assert isinstance(shifted_outer.children[0].node.children[0], PrimitiveNode), "Nested child should be PrimitiveNode"
    assert shifted_outer.children[0].node.children[0].data == 6, "First nested MoveValue should shift to 6"
    assert isinstance(shifted_outer.children[0].node.children[1], PrimitiveNode), "Nested child should be PrimitiveNode"
    assert shifted_outer.children[0].node.children[1].data == 7, "Second nested MoveValue should shift to 7"
    assert isinstance(shifted_outer.children[1], PrimitiveNode), "Outer child should be PrimitiveNode"
    assert shifted_outer.children[1].data == 0, "Outer MoveValue should shift from 7 to 0"

    # Test 10: Original node remains unchanged
    original_node = create_move_node(2)
    shifted_node = shift(original_node, 1)
    assert isinstance(original_node, PrimitiveNode), "Original node should be PrimitiveNode"
    assert original_node.data == 2, "Original node value should remain 2"
    assert isinstance(shifted_node, PrimitiveNode), "Shifted node should be PrimitiveNode"
    assert shifted_node.data == 3, "Shifted node value should be 3"

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
    expected_pos = (RepeatNode(nodes_pos[0], CountValue(3)),)
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
    expected_wrap = (RepeatNode(nodes_wrap[0], CountValue(3)),)
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

def test_find_repeating_pattern():
    """Tests the find_repeating_pattern function for detecting repeating patterns."""
    # Test Case 1: Empty Input
    result = find_repeating_pattern([], 0)
    assert result == (None, 0, False), "Test Case 1 Failed: Empty input should return (None, 0, False)"

    # Test Case 2: Single Node
    node = PrimitiveNode(MoveValue(1))
    result = find_repeating_pattern([node], 0)
    assert result == (None, 0, False), "Test Case 2 Failed: Single node should return (None, 0, False)"

    # Test Case 3: Simple Repeat
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert pattern == nodes[0], "Test Case 3 Failed: Pattern should be MoveValue(2)"
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
    assert isinstance(pattern, ProductNode) and str(pattern) == "20", "Test Case 4 Failed: Pattern should be 20"
    assert count == -3, "Test Case 4 Failed: Count should be -3 for alternating"
    assert is_reversed, "Test Case 4 Failed: Should be reversed"

    # Test Case 5: Multi-Node Pattern
    pattern_nodes = [PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(3))]
    nodes = pattern_nodes * 2  # 2,3,2,3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 0)
    assert isinstance(pattern, ProductNode) and pattern.children == tuple(pattern_nodes), "Test Case 5 Failed: Incorrect pattern"
    assert count == 2, "Test Case 5 Failed: Count should be 2"
    assert not is_reversed, "Test Case 5 Failed: Should not be reversed"

    # Test Case 6: No Repeat
    nodes = [PrimitiveNode(MoveValue(i)) for i in [1, 2, 3]]
    result = find_repeating_pattern(nodes, 0)
    assert result == (None, 0, False), "Test Case 6 Failed: Non-repeating sequence should return (None, 0, False)"

    # Test Case 7: Offset Pattern
    nodes = [PrimitiveNode(MoveValue(1))] + [PrimitiveNode(MoveValue(2))] * 3
    pattern, count, is_reversed = find_repeating_pattern(nodes, 1)
    assert pattern == PrimitiveNode(MoveValue(2)), "Test Case 7 Failed: Pattern should be MoveValue(2)"
    assert count == 3, "Test Case 7 Failed: Count should be 3"
    assert not is_reversed, "Test Case 7 Failed: Should not be reversed"

    print("Test find_repeating_pattern - Passed")

def test_factorize_tuple():
    """Tests the factorize_tuple function for compressing ProductNode and SumNode."""
    # Test Case 1: Empty ProductNode
    node = ProductNode(())
    result = factorize_tuple(node)
    assert result == node, "Test Case 1 Failed: Empty ProductNode should remain unchanged"

    # Test Case 2: Single Node ProductNode
    node = ProductNode((PrimitiveNode(MoveValue(1)),))
    result = factorize_tuple(node)
    assert result == node, "Test Case 2 Failed: Single node ProductNode should remain unchanged"

    # Test Case 3: Repeating ProductNode
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode((RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),))
    assert result == expected, "Test Case 3 Failed: Repeating nodes should be compressed"

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
    expected = ProductNode((RepeatNode(ProductNode((PrimitiveNode(MoveValue(2)), PrimitiveNode(MoveValue(0)))), CountValue(-3)),))
    assert result == expected, "Test Case 4 Failed: Alternating pattern should be compressed with negative count"

    # Test Case 5: Mixed ProductNode
    nodes = [PrimitiveNode(MoveValue(1))] * 3 + [PrimitiveNode(MoveValue(2))]
    product = ProductNode(tuple(nodes))
    result = factorize_tuple(product)
    expected = ProductNode((
        RepeatNode(PrimitiveNode(MoveValue(1)), CountValue(3)),
        PrimitiveNode(MoveValue(2))
    ))
    assert result == expected, "Test Case 5 Failed: Mixed sequence compression incorrect"

    # Test Case 6: SumNode Arithmetic Sequence
    sum_node = SumNode(tuple([PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]]))
    result = factorize_tuple(sum_node)
    expected = SumNode((RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(3)),))
    assert result == expected, "Test Case 6 Failed: SumNode arithmetic sequence should be compressed"

    # Test Case 7: SumNode Non-Sequence
    sum_node = SumNode(tuple([PrimitiveNode(MoveValue(i)) for i in [0, 2, 1]]))
    result = factorize_tuple(sum_node)
    assert result == sum_node, "Test Case 7 Failed: Non-sequence SumNode should remain unchanged"

    # Test Case 8: Non-Product/Sum Node
    node = PrimitiveNode(MoveValue(1))
    result = factorize_tuple(node)
    assert result == node, "Test Case 8 Failed: Non-Product/Sum node should remain unchanged"

    print("Test factorize_tuple - Passed")

def test_factorize_tree():
    """Tests the factorize_tree function for tree-wide compression."""
    # Test Case 1: Empty Tree
    tree = KolmogorovTree(RootNode(ProductNode(()), CoordValue((0, 0)), PaletteValue({1})))
    result = factorize_tree(tree)
    assert result.root == tree.root and result.symbols == tree.symbols, "Test Case 1 Failed: Empty tree should remain unchanged"

    # Test Case 2: Simple Repeating ProductNode
    nodes = [PrimitiveNode(MoveValue(2))] * 3
    root = RootNode(ProductNode(tuple(nodes)), CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree(root)
    result = factorize_tree(tree)
    expected_root = RootNode(
        ProductNode((RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),)),
        CoordValue((0, 0)),
        PaletteValue({1})
    )
    assert result.root == expected_root and result.symbols == [], "Test Case 2 Failed: Repeating ProductNode not compressed"

    # Test Case 3: Tree with Symbols
    symbol = ProductNode((
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2)),
        PrimitiveNode(MoveValue(2))
    ))
    symbol_node = SymbolNode(IndexValue(0), ())
    root = RootNode(ProductNode((symbol_node, PrimitiveNode(MoveValue(3)))), CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree(root, [symbol])
    result = factorize_tree(tree)
    expected_symbol = ProductNode((RepeatNode(PrimitiveNode(MoveValue(2)), CountValue(3)),))
    expected_root = RootNode(
        ProductNode((SymbolNode(IndexValue(0), ()), PrimitiveNode(MoveValue(3)))),
        CoordValue((0, 0)),
        PaletteValue({1})
    )
    assert result.root == expected_root and result.symbols == [expected_symbol], "Test Case 3 Failed: Symbol not factorized"
    # Test Case 4: Nested SumNode
    inner_sum = SumNode(tuple([PrimitiveNode(MoveValue(i)) for i in [0, 1, 2]]))
    root = RootNode(ProductNode((inner_sum, PrimitiveNode(MoveValue(3)))), CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree(root)
    result = factorize_tree(tree)
    expected_inner = SumNode((RepeatNode(PrimitiveNode(MoveValue(0)), CountValue(3)),))
    expected_root = RootNode(
        ProductNode((expected_inner, PrimitiveNode(MoveValue(3)))),
        CoordValue((0, 0)),
        PaletteValue({1})
    )
    assert result.root == expected_root and result.symbols == [], "Test Case 4 Failed: Nested SumNode not compressed"

    print("Test factorize_tree - Passed")

def run_tests():
    """Runs simple tests to verify KolmogorovTree functionality."""
    # Test 1: Basic node creation and bit length
    move_right = PrimitiveNode(MoveValue(2))
    assert move_right.bit_length() == 6, "PrimitiveNode bit length should be 6 (3 + 3)"
    product = ProductNode((move_right, move_right))
    assert product.bit_length() == 15, "ProductNode bit length should be 15 (3 + 6 + 6)"
    move_down = PrimitiveNode(MoveValue(3))
    sum_node = SumNode((move_right, move_down))
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
    assert (move_right | move_down) == SumNode((move_right, move_down)), "Operator | failed"
    assert (move_right & move_down) == ProductNode((move_right, move_down)), "Operator & failed"
    assert (move_right + move_down) == ProductNode((move_right, move_down)), "Operator + failed"
    assert (move_right * 3) == RepeatNode(move_right, CountValue(3)), "Operator * failed"
    assert ((move_right * 2) * 3) == RepeatNode(move_right, CountValue(6)), "Nested * failed"
    print("Test 3: Operator overloads - Passed")

    # Test 4: Symbol resolution
    symbol_def = ProductNode((move_right, move_down))
    symbols: list[KNode[MoveValue]] = [symbol_def]
    symbol_node = SymbolNode(IndexValue(0), ())
    root_with_symbol = RootNode(ProductNode((symbol_node, move_right)), CoordValue((0, 0)), PaletteValue({1}))
    tree = KolmogorovTree[MoveValue](root_with_symbol, symbols)
    resolved = resolve_symbols(tree.root, tree.symbols)
    expected_resolved = RootNode(ProductNode((symbol_def, move_right)), CoordValue((0, 0)), PaletteValue({1}))
    assert resolved == expected_resolved, "Symbol resolution failed"
    print("Test 4: Symbol resolution - Passed")

    test_encode_run_length()
    test_construct_product_node()
    test_shift()
    test_get_iterator()

    test_find_repeating_pattern()
    test_factorize_tuple()
    test_factorize_tree()

if __name__ == "__main__":
    tree = example_usage()
    run_tests()
    print("All tests passed successfully!")
