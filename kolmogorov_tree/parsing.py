"""
Parsing utilities for Kolmogorov Tree string representations.

Functions:
    split_top_level_arguments(s) - Split arguments at top level of brackets
    str_to_repr(s)               - Convert KNode string to Python repr
    str_to_knode(s)              - Parse string to KNode object
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kolmogorov_tree.nodes import KNode


def split_top_level_arguments(s: str) -> list[str]:
    """Splits comma-separated arguments at the outermost bracket level."""
    level = 0
    stack: list[str] = []
    result: list[str] = []

    OPENING_BRACKETS = ["(", "[", "{", "<"]
    CLOSING_BRACKETS = [")", "]", "}", ">"]

    opening_bracket = None
    closing_bracket = None
    delimiter = None

    for char in s:
        if char in OPENING_BRACKETS:
            opening_bracket = char
            closing_bracket = {"(": ")", "[": "]", "{": "}", "<": ">"}[opening_bracket]
            delimiter = {"(": ",", "[": ",", "{": ","}[opening_bracket]
            break

    if opening_bracket is None:
        return []

    for char in s:
        if level > 1:
            stack.append(char)
        if (char == delimiter or char == closing_bracket) and level == 1:
            result.append("".join(stack))
            stack = []
        if level == 1 and char != delimiter:
            stack.append(char)
        if char in OPENING_BRACKETS:
            level += 1
        if char in CLOSING_BRACKETS:
            level -= 1

    return result


def str_to_repr(s: str) -> str:
    """Converts a KNode string representation to Python constructor syntax."""
    s = s.strip()

    if s == "None":
        return "NoneValue(value=None)"

    coord_match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if coord_match:
        x_val, y_val = coord_match.group(1), coord_match.group(2)
        return f"CoordValue(value=Coord(col={x_val}, row={y_val}))"

    palette_match = re.fullmatch(r"set\{\s*(.*?)\s*\}", s)
    if palette_match:
        content = palette_match.group(1).strip()
        if not content:
            colors_repr = ""
        else:
            colors_list = [item.strip() for item in content.split(",")]
            colors_repr = ", ".join(colors_list)
        return f"PaletteValue(value=frozenset({{{colors_repr}}}))"

    if s.startswith("{") and s.endswith("}"):
        content_repr = ", ".join(
            str_to_repr(child) for child in split_top_level_arguments(s)
        )
        return f"SumNode(children=frozenset({{{content_repr}}}))"

    if s.startswith("[") and s.endswith("]"):
        content_repr = ", ".join(
            str_to_repr(child) for child in split_top_level_arguments(s)
        )
        return f"ProductNode(children=({content_repr}))"

    if s.startswith("(") and s.endswith("}"):
        first = split_top_level_arguments(s)[0]
        node = str_to_repr(first)
        rest = s[len(first) + 2 :]
        count = split_top_level_arguments(rest)[0]
        return f"RepeatNode(node={node}, count={count})"

    sym_match = re.fullmatch(r"s_(\d+)\((.*)\)", s)
    if sym_match:
        index_val = sym_match.group(1)
        params_str = sym_match.group(2).strip()
        params_list_repr = []
        if params_str:
            params_list_str = split_top_level_arguments("(" + params_str + ")")
            params_list_repr = [str_to_repr(p) for p in params_list_str]

        index_repr = f"IndexValue(value={index_val})"
        params_tuple_repr = ", ".join(params_list_repr)
        if len(params_list_repr) == 1:
            params_tuple_repr += ","
        return f"SymbolNode(index={index_repr}, parameters=({params_tuple_repr}))"

    if s.startswith("Root(") and s.endswith(")"):
        args = split_top_level_arguments(s)
        if len(args) == 3:
            node_repr = str_to_repr(args[0])
            pos_repr = str_to_repr(args[1])
            colors_repr = str_to_repr(args[2])
            return f"RootNode(node={node_repr}, position={pos_repr}, colors={colors_repr})"

    if s.startswith("Rect(") and s.endswith(")"):
        content_str = s[5:-1]
        args = split_top_level_arguments(content_str)
        if len(args) == 2:
            height_repr = str_to_repr(args[0])
            width_repr = str_to_repr(args[1])
            if height_repr.isdigit():
                height_repr = f"CountValue(value={height_repr})"
            if width_repr.isdigit():
                width_repr = f"CountValue(value={width_repr})"
            return f"RectNode(height={height_repr}, width={width_repr})"

    nested_match = re.fullmatch(r"Y_\{(\d+)\}\((.*)\)", s)
    if nested_match:
        count_val = nested_match.group(1)
        args_str = nested_match.group(2).strip()
        args = split_top_level_arguments(args_str)
        if len(args) == 2:
            index_str, node_str = args[0], args[1]
            count_repr = f"CountValue(value={count_val})"
            index_repr = str_to_repr(index_str)
            if index_repr.isdigit():
                index_repr = f"IndexValue(value={index_repr})"
            node_repr = str_to_repr(node_str)
            return f"NestedNode(index={index_repr}, node={node_repr}, count={count_repr})"

    if s.startswith("Var(") and s.endswith(")"):
        index_str = s[4:-1]
        index_val = int(index_str)
        var_val_repr = f"VariableValue(value={index_val})"
        return f"VariableNode(index={var_val_repr})"

    if s in "01234567":
        return f"PrimitiveNode(value=MoveValue(value={s}))"

    return s


def str_to_knode(s: str) -> "KNode":
    """Parses a string representation into a KNode object."""
    from localtypes import Coord  # noqa: F401

    from kolmogorov_tree.nodes import (  # noqa: F401
        KNode,
        NestedNode,
        PrimitiveNode,
        ProductNode,
        RectNode,
        RepeatNode,
        RootNode,
        SumNode,
        SymbolNode,
        VariableNode,
    )
    from kolmogorov_tree.primitives import (  # noqa: F401
        CoordValue,
        CountValue,
        IndexValue,
        MoveValue,
        NoneValue,
        PaletteValue,
        VariableValue,
    )

    return eval(str_to_repr(s))  # noqa: S307


__all__ = [
    "split_top_level_arguments",
    "str_to_repr",
    "str_to_knode",
]
