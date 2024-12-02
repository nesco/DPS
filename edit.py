"""
This module is dedicated to compute edit distance and transformations.
"""

from enum import StrEnum
from typing import Literal, Optional
from syntax_tree import *

Operation = StrEnum('Operation', ['IDENTITY', 'ADD', 'DELETE', 'SUBSTITUTE'])

# Operation = Literal['I', 'D', 'S']
#TransformationString = tuple[Operation, int, Optional[str], Optional[str]]
TransformationString = tuple[Operation, Optional[str], Optional[str]]
#TransformationNode = tuple
# n -> Repeat(n, cou) Root(s, col, n) (I, len)
# Perform {operation} at {index}: {old_value} -> {new_value}

def transformation_str_to_str(transformation: TransformationString) -> str:
    operation, index, value_old, value_new = transformation
    match operation:
        case "S": return f'Substituting at index {index} : {value_old} by {value_new}'
        case "D": return f'Deleting at index {index} : {value_old}'
        case "I" : return f'Inserting at index {index} : {value_new}'

def edit_distance1(a: str, b: str, reverse=True) -> tuple[int, list[TransformationString]]:
    """
    Edit distance between a and b,
    the transformation is from a to b
    By default the transformations stay in reverse, because
    graph traversals differs the most from the end
    """
    n, m = len(a), len(b)
    dp = [[0] * (m+1) for _ in range(n+1)]

    # First column and row initialization
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # Fill the table
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1

    # Now the edit cost is in dp[n][m]
    # Backtrack to find the steps
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1]:
            i, j = i-1, j-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            steps.append(('S', i-1, a[i-1], b[j-1]))
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            steps.append(('D', i-1, a[i-1], None))
            i -= 1
        else:
            steps.append(('I', j-1, None, b[j-1]))
            j -= 1

    if not reverse:
        steps = steps[::-1]
    return dp[n][m], steps

def edit_distance(a: str, b: str, reverse=True) -> tuple[int, list[TransformationString]]:
    """
    Edit distance between a and b,
    the transformation is from a to b
    By default the transformations stay in reverse, because
    graph traversals differs the most from the end
    """
    n, m = len(a), len(b)
    dp = [[0] * (m+1) for _ in range(n+1)]

    # First column and row initialization
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # Fill the table
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1

    # Now the edit cost is in dp[n][m]
    # Backtrack to find the steps
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1]:
            steps.append((Operation.IDENTITY, i-1, None, None))
            i, j = i-1, j-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            steps.append((Operation.SUBSTITUTE, i-1, a[i-1], b[j-1]))
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            steps.append((Optional.DELETE, i-1, a[i-1], None))
            i -= 1
        else:
            steps.append((Operation.ADD, j-1, None, b[j-1]))
            j -= 1

    if not reverse:
        steps = steps[::-1]
    return dp[n][m], steps

def ast_edit_distance(a: ASTNode, b: ASTNode) -> tuple[int, TransformationNode]:
    match (a, b):
        case (MovesNode, MovesNode):
            distance, transformation_chars: list = edit_distance(a.moves, b.moves)

def distance_moves(a: MovesNode, b: MovesNode) -> tuple[int, list[TransformationString]]:
    return edit_distance(a.moves, b.moves)

a = "kitten"
b = "sitting"

distance, transformations = edit_distance(a, b)

print(f"Distance between {a} and {b}: {distance}")
print(f"Steps: {len(transformations)}")
for transformation in transformations:
    print(transformation_str_to_str(transformation))
