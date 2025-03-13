# Snippets

## Ideas

- "Merging branches": Compress SumNode when they are in ProductNode, to interpret rightfully what there is after as a common thing of all branches:
22\[3,4\]22 == 22\[322,422\] which checks well with the facts that 22\[322,422\] == \[22322, 22422\].
- Previously, for each shape I choose the greedy representation.
A shape S can be represented by (starting_points x traversal_modes) branching chains along their starting points. Those branching chains are usually called "Rose Trees", and with the traversal modes are usually depth-first or breadth-first.
Let's call this set Rose(S). Those chains can be further compressed into what I call "programs", which each as representation of the chains with K-Tree,
which is a pseudo programming language. For every chain C, the set of porgram representing it is KTrees(C), from the literal representation of the RoseTree using ProductNodes and SumNodes
, to more "compressed" versions using RepeatNodes, RectNodes (evil primitive designed specifically for ARC forgive me Father), SymbolNodes along a symbol table.

Note: In fact, for a DF traversal, the order in which the transitions are evaluated matters, thus there are as many "DF traversal" possible as orders of there is transitions
In fact there are more, you could choose to try a possible order at each step, using an based estimator (RL/MCTS or something like model-base optimal control https://latent-planning.github.io) to choose at each point the best possible move


An "minimal" representation of a RoseTree R is the program with the smallest bit length: MKTree: \[RoseTree -> KTree\] S |-> MKTree(S).
All rose trees of a single shape S, are the same length: \exist L \all R \in Rose(S) length(R) == L, with L being the number of pixels in the shape minus - 1: L = #pixels(S) - 1.
However, not all their minimal representations are of the same bit-length: the more "random" in the sense of algorithmic information theory, the longer it will be.
Thus selecting programs with small bit-length select representation that "compress the maximum the information of the shape", thus that reduces it to its fundamentals, exploits it's symetries, etc
There is a greedy approach: try to select the minimal solo-symbolized program (KTree including symbol table), or do some kind of """"cross-entropy""" on the distribution itself: symbolize together all the representations, take the smallest once symbolized *together*

An "objective" representation is obtained by setting a fixed score function over program to

- Repeat(SumNode, count) -> ther SumNode is nested count time. Note that order doesn't matter for a set
- To explore: union as normalized sumnode?
- try several DFs

Bias vs Variance

### How to extract information from a set of things?

To simplify things, let's stay on simple chains:

A: 111001
B: 010101

The purely statistical approach (sometimes called "frequentism") is to simply count the proportions of those programs over the programs of length 6:

111001 -> 1/2
010101 -> 1/2

To think in terms of energy, then the normalization step can be scrapped and it's just counting.

Next step is probabilistic, (sometimes called "bayesian"). If you have any more information, you can add it as a prior:

P(X|D,I) = P(I | X,D) * P(X|D) / P(I) (D: Data, I: prior information)
Where P(I) can be computed as a normalisation constant.

It's basically similar as adding a "model" on the data distribution. You skew from pure variance to something with some bias, which can be more accurate in the low-data regime.
As length is dynamic, and not fixed, the actual family of statistic distributions is more complex

With chains, there is an additional structure than can be exploited: the content and order of symbols<, by introducing new symbols, to improve the efficiency of your model of the data distribution

-> Either by doing sub-statistic over patterns in the chains.
Storing frequent patterns in a lookup table. Here a lookup table can be taken literally, for a compression algorithm, or as the weight of a neural network in the case of a transformer
-> Or by introducing some kind of "language" over the character chain you are outputting: like repetions, nested patterns

Then you can add some kind of abstraction systems, like functions.

The final frontier is to have a meta distribution over the fully-fledge universal programs that output your chains. This is the subject of algorithmic information theory, however it's severely constrained by the halting problem
For example, in this kind of meta distribution, the probability of a program being the "right" one is proportional to it's bit length of its minimal reprensentation, called it's kolmogorov complexity. However it's subject to the halting problem. The choice of programming language is not a too strong bias because it only affects minimal bit length representation up to a constant


When you used compression, like repeats, you add a model over simple statistical counting.
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
