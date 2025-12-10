# Edit Distance Module

This module computes edit distances and transformations between tree-like `BitLengthAware` data structures.

## Two Distance Metrics

### Edit Distance (Thermodynamic)

Uses Landauer's principle: erasing information has irreducible cost.

```python
from edit import edit_distance, symmetric_edit_distance

# Delete costs bit_length (thermodynamic cost)
dist, ops = edit_distance(source, target, symbol_table)
```

| Operation | Cost |
|-----------|------|
| Delete | `element.bit_length()` |
| Add | `element.bit_length()` |
| Substitute | `new_value.bit_length()` |
| Identity | 0 |

**Properties:**
- More symmetric: d(A→B) ≈ d(B→A)
- Good for: structural similarity, correspondence finding

### MDL Distance (Information-Theoretic)

Pure Kolmogorov semantics: receiver knows source, so deletes are free.

```python
from edit import mdl_distance, symmetric_mdl_distance

# Delete costs 0 (receiver already knows what to remove)
dist, ops = mdl_distance(source, target, symbol_table)
```

| Operation | Cost |
|-----------|------|
| Delete | 0 |
| Add | `element.bit_length()` |
| Substitute | `new_value.bit_length()` |
| Identity | 0 |

**Properties:**
- Asymmetric: d(big→small) ≈ 0, d(small→big) = K(additions)
- Good for: "Is B derivable from A?", transformation learning

## When to Use What

| Task | Function | Why |
|------|----------|-----|
| Find correspondences | `symmetric_edit_distance` | Symmetric, penalizes differences |
| Find transformations | `mdl_distance` | Shows what needs to be added |
| Check containment | `mdl_distance` | d(A→B) ≈ 0 means B ⊆ A |
| Score cliques | `joint_complexity` | Rewards shared structure |
| Scale-invariant similarity | `normalized_*_distance` | Range [0,1] |

## API Reference

### Core Distance Functions

```python
from edit import (
    # MDL (zero-cost deletes)
    mdl_distance,           # Returns (dist, operations)
    mdl_distance_value,     # Returns just the distance
    symmetric_mdl_distance, # d(A→B) + d(B→A)

    # Edit (thermodynamic deletes)
    edit_distance,           # Returns (dist, operations)
    edit_distance_value,     # Returns just the distance
    symmetric_edit_distance, # d(A→B) + d(B→A)

    # Low-level (with custom delete_cost_func)
    recursive_edit_distance,
    extended_edit_distance,
)

# Example with custom delete cost
from edit import extended_edit_distance
dist, ops = extended_edit_distance(
    source, target, symbol_table,
    delete_cost_func=lambda x: 0  # MDL semantics
)
```

### Joint Complexity (for Cliques)

```python
from edit import joint_complexity

# K(A,B,C) = K(A) + K(B|A) + K(C|A,B)
# Better than sum of pairwise distances for finding shared structure
score = joint_complexity(
    [elem_a, elem_b, elem_c],
    symbol_table,
    use_mdl=False,  # True for MDL, False for edit distance
)
```

### Normalized Distances

```python
from edit import normalized_information_distance, normalized_compression_distance

# NID: (K(A|B) + K(B|A)) / K(A,B) ∈ [0,1]
nid = normalized_information_distance(a, b, symbol_table, use_mdl=False)

# NCD: (K(A,B) - min(K(A),K(B))) / max(K(A),K(B)) ∈ [0,1+ε]
ncd = normalized_compression_distance(a, b, symbol_table, use_mdl=False)
```

## Operations

### Basic Operations

- **Identity**: No change (cost: 0)
- **Add**: Insert new element (cost: bit_length)
- **Delete**: Remove element (cost: bit_length or 0)
- **Substitute**: Replace value (cost: new_value.bit_length)
- **Inner**: Container for nested field operations

### Extended Operations (Prune/Graft)

Work at **arbitrary depth** in the tree:

```python
# Prune: keep only a descendant at any depth
A → B → C → D   # Prune to D gives: D (removes A,B,C)

# Graft: attach source as descendant of new structure
D   # Graft into A → B gives: A → B → D
```

## Example: ARC Problem

```python
from edit import symmetric_edit_distance, mdl_distance_value, joint_complexity
from kolmogorov_tree import unsymbolize

# For clique finding (which objects correspond across grids)
def correspondence_distance(a, b, symbol_table):
    a_unsym = unsymbolize(a, symbol_table)
    b_unsym = unsymbolize(b, symbol_table)
    return symmetric_edit_distance(a_unsym, b_unsym, symbol_table)

# For transformation learning (input → output)
def transformation_cost(input_obj, output_obj, symbol_table):
    return mdl_distance_value(input_obj, output_obj, symbol_table)

# For scoring cliques
def clique_score(clique_elements, symbol_table):
    elems = [unsymbolize(e, symbol_table) for _, e in clique_elements]
    return joint_complexity(elems, symbol_table, use_mdl=False)
```

## Cost Model Summary

|  | Edit Distance | MDL Distance |
|--|---------------|--------------|
| Delete | `bit_length` | 0 |
| Add | `bit_length` | `bit_length` |
| Substitute | `new.bit_length` | `new.bit_length` |
| Symmetry | ~Symmetric | Asymmetric |
| Use case | Similarity | Transformation |
