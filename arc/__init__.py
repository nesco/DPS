"""
ARC grid encoding/decoding using Kolmogorov syntax trees.

This package provides lossless encoding of ARC grids into abstract syntax trees (ASTs)
that represent shapes using Freeman chain codes.

Submodules:
    arc.types    - Type definitions (Coord, Color, ColorGrid, etc.)
    arc.encoding - Grid-to-AST encoding functions
    arc.decoding - AST-to-grid decoding functions

Example:
    >>> from arc.types import Coord
    >>> from arc.encoding import encode_component
    >>> from arc.decoding import decode_root
    >>>
    >>> component = {Coord(0, 0), Coord(1, 0), Coord(0, 1), Coord(1, 1)}
    >>> colors = {1}
    >>> distribution, symbols = encode_component(component, colors)
    >>> points = decode_root(distribution[0])
"""
