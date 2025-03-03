# Snippets

## Prompts

Description of what I want to do:
"""
I am trying to create a representation language for nondeterministic programs. A programming language is though as a string ,  of a given alphabets, like 8-directional moves for examples for programs that describes the moves in a 2D grid. Non-deterministic programs will thus be represented as a kind of rose tree of possible next sequences (SumNodes/AlternativeNodes being the branches). I will use that to benchmark the solution against ARC AGI, by representing shapes as a non-deterministic programming language using DFS on the connectivity graph.

I want:

1) A representation simple enough so the actual complexity is computable, or at least a close approximate of it's "minimum bit length" / "minimum description length"
2) Expressive enough for this approximate of the minimum description length to also be a good approximate of the actual Kolmogorov complexity

The main idea I had was to:
1. Extract repetions using a RepeatNode
2. Extract patterns, and possibly nested patterns, using memorization / a lookup table of "symbols"
2 bis. Enable lambda abstraction in the same time
"""
+
"""
Any idea of how I should properly code it? A first draft is the kolmogorov_tree.py file attached
"""

->
"""
I am trying to create a representation language for nondeterministic programs. A programming language is though as a string ,  of a given alphabets, like 8-directional moves for examples for programs that describes the moves in a 2D grid. Non-deterministic programs will thus be represented as a kind of rose tree of possible next sequences (SumNodes/AlternativeNodes being the branches). I will use that to benchmark the solution against ARC AGI, by representing shapes as a non-deterministic programming language using DFS on the connectivity graph.

I want:

1) A representation simple enough so the actual complexity is computable, or at least a close approximate of it's "minimum bit length" / "minimum description length"
2) Expressive enough for this approximate of the minimum description length to also be a good approximate of the actual Kolmogorov complexity

The main idea I had was to:
1. Extract repetions using a RepeatNode
2. Extract patterns, and possibly nested patterns, using memorization / a lookup table of "symbols"
2 bis. Enable lambda abstraction in the same time.

Any idea of how I should properly code it? A first draft is the kolmogorov_tree.py file:
```python
from typing import Generic, TypeVar, Callable, Union, Optional, Iterable, Any, List, Set, Dict, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import Counter
import math

# Import RoseTree
from tree import RoseTree, Rose

# Type variables for generic programming
T = TypeVar('T')

class BitLength(IntEnum):
    """Defines bit lengths for various components"""
    NODE_TYPE = 3     # Base length for node type encoding (3 bits for up to 8 types)
    PRIMITIVE = 4     # Length for primitive values (like colors, etc.)
    COUNT = 4         # For repeat counts, typically 2-9 or -2 to -8
    COORD = 10        # For coordinate values
    INDEX = 3         # For indexing, typically < 8 so 3 bits
    VAR = 1           # For variable indices (0 or 1)

@dataclass(frozen=True)
class BitLengthAware(ABC):
    """Protocol for values that know their bit lengths"""

    @abstractmethod
    def bit_length(self) -> int:
        """Return the bit length of this value"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the value"""
        pass

# Base KNode class
@dataclass(frozen=True)
class KNode(Generic[T], ABC):
    """Base abstract class for all nodes in the Kolmogorov Tree"""

    def __len__(self) -> int:
        """Default length is the node type encoding"""
        return BitLength.NODE_TYPE

    def __eq__(self, other) -> bool:
        """Equality based on class and attributes"""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    @abstractmethod
    def bit_length(self) -> int:
        """Calculate the bit length of this node"""
        pass

###############################################################################
# Primitive Types
###############################################################################

@dataclass(frozen=True)
class MoveValue(BitLengthAware):
    """Represents a directional move"""
    direction: int  # 0-7 for 8-connectivity

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE

    def __str__(self) -> str:
        return str(self.direction)

@dataclass(frozen=True)
class MovesValue(BitLengthAware):
    """Represent a list of directional moves"""
    directions: tuple[int]

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE * len(self.direction)

    def __str__(self) -> str:
        return "".join(directions)

@dataclass(frozen=True)
class ColorValue(BitLengthAware):
    """Represents a color value"""
    value: int

    def bit_length(self) -> int:
        return BitLength.PRIMITIVE

    def __str__(self) -> str:
        return f"C{self.value}"

@dataclass(frozen=True)
class CoordValue(BitLengthAware):
    """Represents a coordinate pair"""
    x: int
    y: int

    def bit_length(self) -> int:
        return BitLength.COORD

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

###############################################################################
# Node Types
###############################################################################

# Leaf node
@dataclass(frozen=True)
class PrimitiveNode(KNode[T]):
    """Leaf node containing a primitive BitLengthAware value"""
    value: BitLengthAware

    def bit_length(self) -> int:
        return super().__len__() + self.value.bit_length()

    def __str__(self) -> str:
        return str(self.value)

# Variable node
@dataclass(frozen=True)
class VariableNode(KNode[T]):
    """Represents a variable/placeholder in the program"""
    index: int  # Index of the variable

    def bit_length(self) -> int:
        return super().__len__() + BitLength.VAR

    def __str__(self) -> str:
        return f"Var({self.index})"

# Common parent class for nodes with children
@dataclass(frozen=True)
class SequenceNode(KNode[T], ABC):
    """Abstract base class for nodes that contain multiple children in sequence"""
    children: tuple[KNode[T], ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Convert list to tuple if needed (since we can't directly use default_factory=tuple)
        if isinstance(self.children, list):
            object.__setattr__(self, 'children', tuple(self.children))

# Product node (Sequence)
@dataclass(frozen=True)
class ProductNode(SequenceNode[T]):
    """
    Represents a product type (AND) - sequential evaluation of children
    Corresponds to ConsecutiveNode in the original structure
    """
    def bit_length(self) -> int:
        return super().__len__() + sum(child.bit_length() for child in self.children)

    def __str__(self) -> str:
        return "".join(str(child) for child in self.children)

# Sum node (Alternative)
@dataclass(frozen=True)
class SumNode(SequenceNode[T]):
    """
    Represents a sum type (OR) - one child path is chosen
    Corresponds to AlternativeNode in the original structure
    """
    def bit_length(self) -> int:
        select_bits = math.ceil(math.log2(len(self.children) + 1)) if self.children else 0
        return super().__len__() + select_bits + sum(child.bit_length() for child in self.children)

    def __str__(self) -> str:
        return "[" + "|".join(str(child) for child in self.children) + "]"

# Repeat node
@dataclass(frozen=True)
class RepeatNode(KNode[T]):
    """Represents a repeated node (iteration)"""
    node: KNode[T]
    count: int | VariableNode

    def bit_length(self) -> int:
        count_len = (
            self.count.bit_length() if isinstance(self.count, KNode)
            else BitLength.COUNT
        )
        return super().__len__() + self.node.bit_length() + count_len

    def __str__(self) -> str:
        return f"({str(self.node)})*{{{self.count}}}"

    def __iter__(self):
        if isinstance(self.count, VariableNode):
            raise ValueError("Cannot iterate a repeat with a variable count")
        return (self.node for _ in range(self.count))

# Symbol node
@dataclass(frozen=True)
class SymbolNode(KNode[T]):
    """
    Represents a symbolic reference (abstraction)
    """
    index: int
    parameters: tuple[BitLengthAware, ...] = field(default_factory=tuple)
    referenced_length: int = 0  # Bit length of the referenced template

    def bit_length(self) -> int:
        params_len = sum(param.bit_length() for param in self.parameters)
        return super().__len__() + BitLength.INDEX + params_len

    def __str__(self) -> str:
        if self.parameters:
            return f"s_{self.index}({', '.join(str(p) for p in self.parameters)})"
        return f"s_{self.index}"

# Root node
@dataclass(frozen=True)
class RootNode(KNode[T]):
    """Represents the root of an ARC AGI 2D path program with position and color set"""
    position: tuple[int, int] | VariableNode
    colors: set[int] | VariableNode
    node: KNode[T] | None = None

    def bit_length(self) -> int:
        pos_len = (
            self.position.bit_length() if isinstance(self.position, KNode)
            else BitLength.COORD
        )
        colors_len = (
            self.colors.bit_length() if isinstance(self.colors, KNode)
            else BitLength.PRIMITIVE * len(self.colors)
        )
        node_len = self.node.bit_length() if self.node else 0
        return super().__len__() + pos_len + colors_len + node_len

    def __str__(self) -> str:
        node_str = str(self.node) if self.node else ""
        return f"{self.context}->{self.position}:{node_str}"

###############################################################################
# KolmogorovTree - Main class
###############################################################################

@dataclass
class KolmogorovTree:
    """
    Represents a complete Kolmogorov program with a root node and symbol table
    """
    root: KNode
    symbols: list[KNode] = field(default_factory=list)

    def bit_length(self) -> int:
        """Calculate the total bit length of this program"""
        return self.root.bit_length() + sum(symbol.bit_length() for symbol in self.symbols)

    def resolve_symbols(self) -> KNode:
        """Resolve all symbolic references to create a fully expanded tree"""
        return resolve_symbols(self.root, self.symbols)

    def __str__(self) -> str:
        symbols_str = "\n".join(f"s_{i}: {str(s)}" for i, s in enumerate(self.symbols))
        return f"Symbols:\n{symbols_str}\n\nProgram:\n{str(self.root)}"

    def kolmogorov_complexity(self) -> int:
        """
        Calculate the Kolmogorov complexity (approximated by bit length)
        of the program represented by this tree
        """
        return self.bit_length()

###############################################################################
# Helper functions
###############################################################################

def resolve_symbols(node: KNode, symbols: list[KNode]) -> KNode:
    """Resolve symbolic references in a node using the symbol table"""

    def replace_variables(template: KNode, params: tuple[Any, ...]) -> KNode:
        """Replace variables in a template with provided parameters"""
        if isinstance(template, VariableNode) and template.index < len(params):
            param = params[template.index]
            return param if isinstance(param, KNode) else PrimitiveNode(param)

        if isinstance(template, ProductNode):
            return ProductNode([replace_variables(child, params) for child in template.children])

        if isinstance(template, SumNode):
            return SumNode([replace_variables(child, params) for child in template.children])

        if isinstance(template, RepeatNode):
            new_node = replace_variables(template.node, params)
            new_count = replace_variables(template.count, params) if isinstance(template.count, KNode) else template.count
            return RepeatNode(new_node, new_count)

        if isinstance(template, SymbolNode):
            # Handle nested symbols
            new_params = tuple(
                replace_variables(p, params) if isinstance(p, KNode) else p
                for p in template.parameters
            )
            return SymbolNode(template.index, new_params, template.referenced_length)

        if isinstance(template, RootNode):
            new_pos = replace_variables(template.position, params) if isinstance(template.position, KNode) else template.position
            new_ctx = replace_variables(template.context, params) if isinstance(template.context, KNode) else template.context
            new_node = replace_variables(template.node, params) if template.node else None
            return RootNode(new_pos, new_ctx, new_node)

        # PrimitiveNode or other types
        return template

    # Handle symbolic references
    if isinstance(node, SymbolNode):
        if 0 <= node.index < len(symbols):
            template = symbols[node.index]
            # Apply parameters to the template
            return replace_variables(template, node.parameters)
        return node  # Invalid symbol index

    # Recursively resolve children
    if isinstance(node, ProductNode):
        return ProductNode([resolve_symbols(child, symbols) for child in node.children])

    if isinstance(node, SumNode):
        return SumNode([resolve_symbols(child, symbols) for child in node.children])

    if isinstance(node, RepeatNode):
        new_node = resolve_symbols(node.node, symbols)
        new_count = resolve_symbols(node.count, symbols) if isinstance(node.count, KNode) else node.count
        return RepeatNode(new_node, new_count)

    if isinstance(node, RootNode):
        new_pos = resolve_symbols(node.position, symbols) if isinstance(node.position, KNode) else node.position
        new_ctx = resolve_symbols(node.context, symbols) if isinstance(node.context, KNode) else node.context
        new_node = resolve_symbols(node.node, symbols) if node.node else None
        return RootNode(new_pos, new_ctx, new_node)

    # PrimitiveNode, VariableNode or other types without children
    return node

def find_common_patterns(trees: list[KolmogorovTree], min_occurrences=2, max_patterns=10) -> list[KNode]:
    """
    Find common patterns across multiple KolmogorovTrees
    Returns a list of nodes that appear frequently
    """
    # Collect all subtrees
    all_subtrees = []

    def collect_subtrees(node, collection):
        # Add this node
        collection.append(node)

        # Process children based on node type
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

    # Collect subtrees from all trees
    for tree in trees:
        collect_subtrees(tree.root, all_subtrees)
        for symbol in tree.symbols:
            collect_subtrees(symbol, all_subtrees)

    # Count occurrences using string representation
    counter = Counter(str(node) for node in all_subtrees)

    # Find nodes that appear frequently
    common_nodes = []
    for node in all_subtrees:
        node_str = str(node)
        if (counter[node_str] >= min_occurrences and
            node_str not in [str(n) for n in common_nodes]):
            common_nodes.append(node)

    # Sort by complexity and frequency
    common_nodes.sort(key=lambda n: (-counter[str(n)], -n.bit_length()))

    # Return top patterns
    return common_nodes[:max_patterns]

def symbolize(trees: list[KolmogorovTree]) -> KolmogorovTree:
    """
    Create a new KolmogorovTree with common patterns extracted as symbols
    """
    # Find common patterns
    common_patterns = find_common_patterns(trees)

    # Create a new symbol table
    symbols = list(common_patterns)

    # Replace patterns in each tree
    symbolized_trees = []
    for tree in trees:
        # Process a node and replace common patterns with symbols
        def process_node(node):
            # First check if this node matches a common pattern
            for i, pattern in enumerate(common_patterns):
                if str(node) == str(pattern):
                    return SymbolNode(i, (), pattern.bit_length())

            # If no match, recursively process children
            if isinstance(node, ProductNode):
                return ProductNode([process_node(child) for child in node.children])

            if isinstance(node, SumNode):
                return SumNode([process_node(child) for child in node.children])

            if isinstance(node, RepeatNode):
                new_node = process_node(node.node)
                return RepeatNode(new_node, node.count)

            if isinstance(node, RootNode):
                new_node = process_node(node.node) if node.node else None
                return RootNode(node.position, node.context, new_node)

            # No changes for other node types
            return node

        # Process the tree
        new_root = process_node(tree.root)
        symbolized_trees.append(KolmogorovTree(new_root, symbols))

    # Return the first symbolized tree (or a merged tree if needed)
    if symbolized_trees:
        return symbolized_trees[0]
    return KolmogorovTree(ProductNode([]))

###############################################################################
# Factory functions for creating common patterns
###############################################################################

def create_move_node(direction: int) -> PrimitiveNode:
    """Create a node representing a move in a specific direction"""
    return PrimitiveNode(MoveValue(direction))

def create_color_node(color: int) -> PrimitiveNode:
    """Create a node representing a color"""
    return PrimitiveNode(ColorValue(color))

def create_coord_node(x: int, y: int) -> PrimitiveNode:
    """Create a node representing a coordinate"""
    return PrimitiveNode(CoordValue(x, y))

def create_moves_sequence(directions: str) -> KNode:
    """Create a sequence of move nodes from a string of directions"""
    if not directions:
        return ProductNode([])

    # Convert string to nodes
    moves = [create_move_node(int(d)) for d in directions]

    # Look for repeating patterns
    if len(moves) >= 3:
        for pattern_len in range(1, len(moves) // 2 + 1):
            if len(moves) % pattern_len == 0:
                pattern = moves[:pattern_len]
                repeat_count = len(moves) // pattern_len

                # Check if the sequence is made of repeated patterns
                is_repeating = True
                for i in range(repeat_count):
                    segment = moves[i*pattern_len:(i+1)*pattern_len]
                    if segment != pattern:
                        is_repeating = False
                        break

                if is_repeating and repeat_count > 1:
                    return RepeatNode(ProductNode(pattern), repeat_count)

    return ProductNode(moves)

def create_rect(height: int, width: int) -> KNode:
    """Create a node representing a rectangle of the given dimensions"""
    if height < 2 or width < 2:
        return ProductNode([])

    # Generate moves to draw a rectangle
    # First row: move right (width-1) times
    first_row = "2" * (width - 1)

    # Subsequent rows: move down, then move left or right (width-1) times
    other_rows = ""
    for i in range(1, height):
        direction = "0" if i % 2 else "2"  # Alternate between left and right
        other_rows += "3" + direction * (width - 1)

    return create_moves_sequence(first_row + other_rows)

###############################################################################
# Example usage
###############################################################################

def example_usage():
    """Example of using the KolmogorovTree"""

    # Create some basic nodes
    move_right = create_move_node(2)
    move_down = create_move_node(3)

    # Create a product node (sequence)
    sequence = ProductNode([move_right, move_right, move_down])

    # Create a repeat node
    repeat = RepeatNode(ProductNode([move_right, move_down]), 3)

    # Create a sum node (alternative)
    alternative = SumNode([sequence, repeat])

    # Create a root node with position and context
    root = RootNode((0, 0), {1}, alternative)

    # Create a KolmogorovTree
    tree = KolmogorovTree(root)

    # Calculate its complexity
    complexity = tree.kolmogorov_complexity()
    print(f"Kolmogorov complexity: {complexity} bits")

    return tree

if __name__ == "__main__":
    example_usage()
```

In addition, here is what ChatGPT gave me about more literature on the subject if it can help:
"""
 will look into intermediate languages that are not Turing complete but are still sufficiently expressive to encode computational processes, particularly for compressing strings. This includes investigating models such as finite-state transducers, combinator-based languages, and other formal systems that allow for structured transformations without the complexity of full Turing-completeness. I’ll update you once I have the findings.

Research completed in 13m· 16 sources
Non-Turing Complete Languages for Structured String Computation

Designing an intermediate language that is not Turing-complete yet expressive enough for structured computations (like string compression) is an active area of research. The goal is to allow complex transformations on strings while avoiding the pitfalls of full Turing completeness (such as non-termination and uncomputable Kolmogorov complexity). By restricting computational power, we often gain decidability and even efficient algorithms for tasks that are intractable in the general case of arbitrary programs​
PEOPLE.CSAIL.MIT.EDU
. Below we explore several formal models that fit these criteria, including finite-state transducers, grammar or combinator-based languages, and other restricted frameworks. We discuss how each can perform structured transformations, how Kolmogorov complexity is defined in each (and whether it’s computable), plus examples, use cases, and known implementations.
Finite-State Transducers and Automatic Complexity

Finite-State Transducers (FSTs) are like finite automata that produce output. They have a finite set of states and transitions that consume input symbols (or epsilon) and emit output symbols. Since they have no unbounded memory (only a finite state), they are not Turing complete (they recognize/transform at most regular languages and rational relations). Despite this, they can perform non-trivial structured transformations on strings in a well-defined way. For example, an FST could be constructed to perform run-length encoding (to the extent of its finite memory) or to substitute patterns in text. In practice, FSTs are widely used in domains like natural language processing (for morphological analysis, phonological rules, etc.) and digital circuits – any setting where a simple, guaranteed-terminating transformation is needed.
Kolmogorov Complexity with FSTs: By using FSTs as the description model instead of Turing machines, we get a computable analogue of Kolmogorov complexity. Finite-state complexity is defined as the length of the shortest description (T, p) where T is a finite-state transducer and p an input such that T(p) = x (produces the string x)​
CS.AUCKLAND.AC.NZ
​
CS.AUCKLAND.AC.NZ
. In other words, it’s the length of the smallest FST (plus its input) that generates a given string. This measure was introduced as a “computable counterpart to Kolmogorov complexity”​
CS.AUCKLAND.AC.NZ
, because the space of all finite transducers can be effectively enumerated​
CS.AUCKLAND.AC.NZ
. Unlike standard Kolmogorov complexity (uncomputable), finite-state complexity is actually computable​
CS.AUCKLAND.AC.NZ
 – one can in principle search through all FSTs up to a certain size. In fact, researchers have developed algorithms to calculate or approximate finite-state complexity for strings​
CS.AUCKLAND.AC.NZ
​
CS.AUCKLAND.AC.NZ
. For example, Roblot (2019) provides an algorithm to compute finite-state complexity and uses it to analyze DNA sequences​
CS.AUCKLAND.AC.NZ
.
Expressiveness and Limitations: Finite-state transducers can capture local and regular patterns in strings. They excel at transformations like prefix/suffix additions, fixed-format encoding, or compressing simple repetitive sequences. For instance, a transducer with a small loop can output "AB" repeatedly, compressing a long repetition of "ABABAB..." into a cycle in the machine. If a string has a periodic structure or repeated substrings, a carefully designed FST can generate it with far fewer states than the string’s length (effectively compressing it). However, because an FST has no stack or unbounded counter, it cannot easily handle nested or arithmetic-dependent patterns beyond a fixed scope. For example, an FST cannot enforce an arbitrary length count or generate truly nested matching pairs – those require more memory (see pushdown models below). In practical compression, a finite-state model might be too weak to capture all redundancy unless combined with extra hints (like an advice input encoding a count). Research on “automatic Kolmogorov complexity” (Kolmogorov complexity with automata-based descriptions) confirms that restricting to finite automata loses some power but still allows meaningful compression measures​
PEOPLE.CSAIL.MIT.EDU
​
PEOPLE.CSAIL.MIT.EDU
.
Use Cases and Implementations: Finite-state transducers are used wherever a guaranteed-terminating, efficient string transformation is needed. In compilers and data processing, FST-like regex replacements or tokenizers perform structured edits without full computation. In compression theory, finite-state compressors (like certain dictionary coders) operate in streaming fashion with finite memory. The concept of finite-state complexity has been implemented in research prototypes – for example, by enumerating transducers to find minimal descriptions​
CS.AUCKLAND.AC.NZ
. While not a mainstream tool like ZIP or PNG, these studies provide theoretical frameworks and algorithms for using FSTs as a compression language. An interesting outcome of such a restricted model is that Kolmogorov complexity becomes decidable: if all programs always halt, one can brute-force search for the shortest program that produces a given string​
CS.STACKEXCHANGE.COM
. (Of course, the search might be expensive, but it will terminate with an answer, unlike the general case.) In summary, finite-state transducers offer a regular, loop-free way to represent string mappings, making complexity measures tractable and transformations efficient for certain classes of patterns.
Grammar-Based Combinator Languages (Context-Free Models)

Another rich approach is using combinator-based languages, notably grammar formalisms, to describe strings. Context-free grammars (CFGs) are a classic example: they use production rules as combinators to build strings from smaller parts. If we restrict a grammar to produce exactly one string (for example, a straight-line grammar with no branching recursion), it becomes a description of that specific string – essentially a compressed form. Grammars are not Turing complete; they can generate potentially infinite languages, but they cannot perform arbitrary computation or simulate a Turing machine’s behavior. This makes them a good intermediate language for structured data: they capture hierarchical patterns (like repetition and self-similarity) but remain decidable and more tractable than general programs.
Kolmogorov Complexity with Grammars: The length of the smallest context-free grammar that generates a string is a well-studied proxy for Kolmogorov complexity. It’s been called the “grammar complexity” of a string​
PEOPLE.CSAIL.MIT.EDU
. This measure is still difficult to compute exactly (finding the minimal grammar is NP-hard in general), but it is decidable (one could brute-force or use approximations) and more constrained than full Kolmogorov complexity. In fact, “weakening the model from Turing machines to context-free grammars reduces the complexity of the problem from the realm of undecidability to mere intractability”​
PEOPLE.CSAIL.MIT.EDU
. In other words, we trade an uncomputable problem for a computable (though possibly NP-hard) one. Research by Charikar et al. (2005) explicitly makes this connection, calling the smallest grammar a “natural but more tractable variant of Kolmogorov complexity”​
PEOPLE.CSAIL.MIT.EDU
. They also showed that while finding the perfect minimal grammar is hard, it’s approximable within $O(\log^2 n)$ factors with algorithms, and many practical compression schemes are essentially grammar-based​
PEOPLE.CSAIL.MIT.EDU
​
PEOPLE.CSAIL.MIT.EDU
.
Expressiveness: Grammar-based languages can express hierarchical and nested structure that finite-state models cannot. A grammar can have nonterminal symbols that stand for recurring substrings or patterns. For example, consider the string: ABABABAB. A context-free grammar can capture its regularity succinctly:
S -> X X X X
X -> AB
This grammar uses a nonterminal X to represent the substring "AB", and the start symbol S concatenates four copies of X to generate the full string. The grammar (with 2 rules) is a much shorter description than writing out ABABABAB. Similarly, grammars can handle nested patterns: e.g., a rule like S -> "(" S ")" (with a base case) can generate balanced parentheses of arbitrary depth. For a specific string with nested structure, a straight-line grammar can replicate that pattern without having to explicitly spell out each level. Essentially, grammar rules act like combinators that build larger structures from smaller ones (concatenation, alternation, etc.), giving a structured compression.
Examples and Use Cases: Grammar-based compression is an active practical field. Algorithms like Sequitur, Re-Pair, and others build a context-free grammar for a given string as a compressed representation. For instance, Sequitur (Nevill-Manning & Witten) incrementally constructs rules by identifying repeated substrings and replacing them with nonterminals. The result is a straight-line grammar (no recursion or branching) that produces exactly the input string. Such a grammar can be viewed as a program in a restricted language: the “operations” are rule expansions (a form of combinator). This yields a structured form of the string that often reveals its regularities. In practice, grammar compression has been used for universal data compression (it can adapt to various data sources) and for pattern discovery. Notably, finding a small grammar helps identify meaningful patterns in data (e.g., motifs in DNA or repeated phrases in text)​
PEOPLE.CSAIL.MIT.EDU
. A string compressed as a grammar is also more interpretable than one compressed as, say, arbitrary bit codes – one can read the grammar rules to see the repetitive or nested structure​
PEOPLE.CSAIL.MIT.EDU
. Because of this, grammar-based methods have been used in areas like DNA sequence analysis, music analysis, and natural language, where understanding the structure is as important as compression​
PEOPLE.CSAIL.MIT.EDU
.
Theoretical Frameworks: The “smallest grammar problem” is the formal task of finding the minimum-size grammar for a string. It has ties to Kolmogorov complexity (with grammars as the model) and has spurred theoretical work on approximation algorithms​
PEOPLE.CSAIL.MIT.EDU
. Researchers have also considered generalizations like non-deterministic grammars or grammars with an advice string (where the grammar can have a choice and an extra input to guide choices)​
PEOPLE.CSAIL.MIT.EDU
​
PEOPLE.CSAIL.MIT.EDU
. Interestingly, even these more powerful grammar models remain not much more efficient in description size than normal straight-line grammars​
PEOPLE.CSAIL.MIT.EDU
​
PEOPLE.CSAIL.MIT.EDU
 – there are theoretical results showing they’re equivalent up to constant factors​
PEOPLE.CSAIL.MIT.EDU
. This suggests grammar-based description is a robust framework. In summary, combinator languages like grammars allow structured, hierarchical string generation without full computational power, yielding a well-defined (if NP-hard) complexity measure that approximates Kolmogorov complexity in a more manageable way​
PEOPLE.CSAIL.MIT.EDU
. Numerous compression tools and research prototypes implement this approach (Sequitur, Re-Pair, etc. for general data; specialized CFG inference in DNA and linguistics; and even pattern-search algorithms that operate on grammar-compressed data).
Bounded or Total Computation Languages (Loop/Combinator Languages)

Beyond automata and grammars, one can design programming languages that allow arithmetic and more general operations but stop short of Turing completeness. Typically, this is done by forbidding unbounded loops or recursion. These languages can perform structured computations on strings and numbers (even more flexibly than a grammar can) but will always terminate, making various analyses decidable. An example is the classic theoretical LOOP language in computability theory, which allows assignment and bounded for-loops but no while-loops or recursion. Such a language can compute any primitive recursive function, but cannot diverge or simulate full Turing machines. Another example is any total functional language (like total subsets of Haskell/ML or Coq’s Gallina) where all recursion must be well-founded (guaranteed to terminate).
Kolmogorov Complexity in a Total Language: If we measure Kolmogorov complexity with respect to a language that is not Turing-complete (but still universal for total computations), the complexity becomes in principle computable. The reason is that we can enumerate all programs in order of size and run them, and we know none will hang indefinitely (they always halt with some output)​
CS.STACKEXCHANGE.COM
. Therefore, given a string w, one can eventually find the shortest program in this language that outputs w by exhaustive search​
CS.STACKEXCHANGE.COM
. In the worst case, this search might be exponential, but it is a decidable procedure. In fact, for some very restricted languages, there are efficient algorithms to compute the complexity. For instance, consider a toy language that can only set variables to constants, double them, halve them, and output literal characters or repeat a block a fixed number of times (similar to the language described in the StackExchange post). In that simple language, the only compression mechanism is detecting repeated substrings. Indeed, the answer to that question notes you can compute the Kolmogorov complexity in polynomial time for that language by dynamic programming, essentially finding repetitive patterns​
CS.STACKEXCHANGE.COM
. Generally, the more we restrict the operations, the easier it is to compute the minimal program.
Expressiveness and Examples: A bounded language with loops can compress strings by exploiting simple regularities like repetition or symmetry. For example, imagine a language with a loop construct repeat n { ... } that executes a block a fixed number of times. A program in this language to output AAAAAA... (100 A’s) might look like n=100; repeat n { print "A"; }. This program is much shorter than printing each A individually. Unlike an FST, this language can use a number variable to represent the count (here 100) and a looping combinator to repeat output. As long as n is a fixed value in the program (not an input that could grow arbitrarily), the language remains non-universal. Another example: a language might allow defining a macro or subroutine and calling it multiple times. If recursion is not allowed (or is bounded), the macro acts like a grammar nonterminal – it can expand to a fixed string pattern wherever invoked. This is essentially a macro substitution compressor (a form of grammar). Many “combinator-based” descriptions can be viewed this way. For instance, the Lempel-Ziv (LZ77/78) compression can be seen as a program that says “at this point, copy the substring that appeared X characters ago of length Y”. That is not a full language, but it’s a fixed schema of two combinators: literal output and back-reference. LZ compression is not Turing-complete (it can’t, say, loop arbitrarily without eventually consuming input), yet it’s expressive enough to capture a wide range of redundancies. In fact, it’s known that standard compressors like LZ or bzip2 do not embed a universal computation; for example, the bzip2 decompressor is not Turing complete​
STACKOVERFLOW.COM
 – it’s a finite algorithm working within bounded memory and rules.
Use Cases: These bounded languages are mainly of theoretical interest in measuring information content, but they also appear in practice as domain-specific languages. For instance, smart contract languages like Ethereum’s Solidity are Turing-complete, but others (like Clarity for Stacks blockchain) were deliberately made not Turing-complete (no unbounded loops) to ensure decidability and safety​
NEWS.YCOMBINATOR.COM
. In data engineering, query languages such as SQL or Datalog disallow infinite loops or recursion by design (Datalog, for example, is not Turing-complete and can only express PTIME queries). These languages show that we can still perform useful structured computations under a termination guarantee. For string processing and compression specifically, one could imagine a scriptable compressor that uses a limited instruction set (like moves, copies, loops with immediate bounds) to build the output. While not common as user-facing languages, several compression algorithms effectively define a mini-language of their own (e.g., regex-based find/replace scripts, or the sequence of copy commands in LZ decompression). If one formally fixes such a language, we obtain a framework where Kolmogorov complexity is well-defined and we could attempt to compute it.
Summary: By choosing an intermediate language that is not Turing-complete – be it a finite-state transducer, a context-free grammar, or a bounded-loop program – we impose enough structure that every “program” (description) halts and its size can be used as a description length. This yields a structured Kolmogorov complexity that is often computable or at least approximable. Finite-state transducers give us automatic complexity that’s computable and grounded in regular transformations​
CS.AUCKLAND.AC.NZ
​
CS.AUCKLAND.AC.NZ
. Grammar-based descriptions give a hierarchy of grammar complexities that connect to Kolmogorov complexity while remaining within reach of algorithms​
PEOPLE.CSAIL.MIT.EDU
. Combinator and loop languages provide a way to incorporate arithmetic or more complex operations but still avoid the halting problem, making brute-force search for minimal programs possible​
CS.STACKEXCHANGE.COM
. Each model comes with trade-offs in what patterns it can compress (regular versus hierarchical versus arithmetic repetition), but all serve as examples that useful computation does not require Turing completeness. These restricted models are not just toy theories; they underpin practical compression tools and have dedicated research frameworks for computing description complexity in a structured, efficient way.References:
Roblot, The Computation of Finite-State Complexity – defines a computable Kolmogorov-like complexity using finite-state transducers​
CS.AUCKLAND.AC.NZ
​
CS.AUCKLAND.AC.NZ
.
Shen et al., Automatic Kolmogorov Complexity and Normality – discusses description modes restricted to finite automata (automatic relations) and their properties.
Charikar et al., Approximating the Smallest Grammar – connects smallest CFG problems to Kolmogorov complexity and gives approximation algorithms​
PEOPLE.CSAIL.MIT.EDU
​
PEOPLE.CSAIL.MIT.EDU
.
Nevill-Manning & Witten, Identifying Hierarchical Structure in Sequences (Sequitur algorithm) – a practical grammar-based compressor for strings.
StackExchange (cs.SE) discussion on Kolmogorov complexity in a non-Turing-complete language – illustrates how having no infinite loops makes complexity decidable and even polynomial-time computable for a simple language​
CS.STACKEXCHANGE.COM
.
""""
"""



Improvement of kolmogorov_tree.py:
"""
I am trying to create a generic structure, that will be used to cleanly create the AST of  'syntax_tree.py'. I will call it KolmogorovTree. The idea is it encodes a non-deterministic "program" (represented initially by branches of tree with data being ‘string-like’ for example or list-like) .It stays in a simplified space, contrarily to a full programming language / Turing machine, the bit length CAN be evaluated. Especially if I put the primitives as a « BitLengthAware » type. The first draft can be found in 'kolmogorov_tree.py'. Can you give me a master plan on how to create the full module clean kolmogorov, including tests, and then to simplify 'syntax_tree.py' by making use of it?
"""
