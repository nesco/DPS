# Snippets

## Prompts

Improvement of kolmogorov_tree.py:
"""
I am trying to create a generic structure, that will be used to cleanly create the AST of  'syntax_tree.py'. I will call it KolmogorovTree. The idea is it encodes a non-deterministic "program" (represented initially by branches of tree with data being ‘string-like’ for example or list-like) .It stays in a simplified space, contrarily to a full programming language / Turing machine, the bit length CAN be evaluated. Especially if I put the primitives as a « BitLengthAware » type. The first draft can be found in 'kolmogorov_tree.py'. Can you give me a master plan on how to create the full module clean kolmogorov, including tests, and then to simplify 'syntax_tree.py' by making use of it?
"""
