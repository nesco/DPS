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

## Design Limitations

### Subtree Operations vs Node Operations

This implementation uses a **subtree-based** edit model, not the classic **node-based**
tree edit distance (Zhang-Shasha, APTED). The semantic models differ:

| Operation | Classic Tree Edit | This Implementation |
|-----------|-------------------|---------------------|
| Insert node between parent/child | ✓ Relinks children to new node | ✗ Not supported |
| Delete middle node | ✓ Relinks children to grandparent | ✗ Not supported |
| Substitute | Relabels single node | Replaces entire subtree |
| Extract subtree | Manual | ✓ Prune operation |
| Wrap in ancestor | Manual | ✓ Graft operation |

**Classic "insert between":**
```
Before:  A → C
After:   A → B → C   (B inserted, C relinked as child of B)
```

**This implementation cannot express relinking.** It operates on whole subtrees:
- `Prune`: Extract a descendant subtree, discarding ancestors
- `Graft`: Wrap source subtree in new ancestor structure
- `Add/Delete`: Only for collection elements (tuples, frozensets)

**When this matters:**
- Syntax tree diffing → Use Zhang-Shasha or APTED instead
- AST transformations → Classic model better for node-level edits

**When this is fine:**
- MDL/Kolmogorov complexity measurement
- Frozen dataclass trees (natural fit for subtree semantics)
- Finding structural correspondences between trees

### Greedy Extended Edit Distance

The `extended_edit_distance` algorithm uses a **greedy heuristic**, not globally
optimal dynamic programming:

1. Compute `recursive_edit_distance(source, target)` as baseline
2. Try `Prune(source→D) + distance(D, target)` for each descendant D of source
3. Try `distance(source, D) + Graft(D→target)` for each descendant D of target
4. Return minimum

**Limitation:** Prune/Graft are only considered at the root level. The recursive
calls use `recursive_edit_distance` which doesn't consider Prune/Graft for internal
nodes.

**Example where suboptimal:**
```python
# Source: nested wrappers around X and Y
source = Outer(Inner(X), Y)
# Target: X directly under Outer, Z replaces Y
target = Outer(X, Z)

# Optimal: Prune Inner→X at internal level, Substitute Y→Z
# Algorithm may not find this if Prune benefit is internal
```

**Complexity trade-off:**
- Current: O(n²) for descendant enumeration — practical for large trees
- Optimal with Prune/Graft at all levels: O(n⁴) or worse

**When this matters:**
- Deep nesting where optimal involves internal Prune/Graft
- Guaranteed-optimal edit scripts required

**When this is fine:**
- Most practical use cases (heuristic finds good solutions)
- Performance-sensitive applications
- Approximate distances are acceptable

## Metric Selection Guide

### For Cliques / Category Discovery

**Problem with raw distances:** Raw edit costs scale with description length. Large objects
appear far from everything; small objects cluster together. This produces "size buckets"
rather than semantic categories.

**Problem with directional MDL:** Forward MDL `K(B|A)` measures containment ("bits to specify
B given A"). It's asymmetric, which breaks reciprocal nearest neighbor algorithms.

**Recommended metrics for cliques:**

| Metric | Function | Use Case |
|--------|----------|----------|
| NID (best for AIT) | `normalized_information_distance(..., metric="mdl")` | Information-theoretic similarity |
| Structural | `structural_distance_value(...)` | Ignore position/color variations |
| Symmetric Edit | `symmetric_edit_distance(...)` | Structural difference magnitude |

```python
# For clique finding - ALWAYS use symmetric metrics
from edit import normalized_information_distance, symmetric_edit_distance

# Good: symmetric, normalized
dist = normalized_information_distance(a, b, symbol_table, metric="mdl")

# Good: symmetric
dist = symmetric_edit_distance(a, b, symbol_table)

# BAD for cliques: directional!
dist = extended_edit_distance(a, b, symbol_table)[0]  # Don't use for cliques
```

### For Morphisms / Transformations

**Use directional MDL** to measure "how much extra description needed to get output from input":

```python
from edit import mdl_distance

# Forward cost: K(output | input)
dist, ops = mdl_distance(input_obj, output_obj, symbol_table)
```

**When to prefer edit distance for morphisms:**

If you want to penalize information destruction (discourage "delete most and rebuild"),
use thermodynamic edit distance as a regularizer:

```python
from edit import mdl_distance, edit_distance

# Propose morphisms with MDL (what needs to be added)
mdl_cost, ops = mdl_distance(input_obj, output_obj, symbol_table)

# Regularize with edit distance (penalize destructive rewrites)
edit_cost, _ = edit_distance(input_obj, output_obj, symbol_table)
total_cost = mdl_cost + lambda_reg * edit_cost
```

### Multiple Representations / Settings

If objects can be encoded under multiple decompositions (connectivity modes, symbolization
choices, repeat extraction thresholds), distances between *objects* should be distances
between *sets of candidate trees*.

**Options:**

1. **Min-over-representations** (can "cheat" with weird encodings):
   ```
   d(X,Y) = min_{x ∈ R(X), y ∈ R(Y)} d(x,y)
   ```

2. **MDL-regularized min** (penalizes unusual encodings):
   ```
   d(X,Y) = min_{x,y} (L(x) + L(y) + d(x,y))
   ```
   where `L(·)` is encoding cost (bit_length).

3. **EM-like approach:** Pick representation per object that minimizes total clique score.
   The `joint_complexity` function supports this pattern.

### Summary Table

| Task | Metric | Function |
|------|--------|----------|
| Clique discovery | NID | `normalized_information_distance(..., metric="mdl")` |
| Clique discovery | Structural | `structural_distance_value(...)` |
| Clique discovery | Edit-based | `symmetric_edit_distance(...)` |
| Morphism learning | MDL | `mdl_distance(input, output, ...)` |
| Morphism regularization | Edit | `edit_distance(input, output, ...)` |
| Size-invariant similarity | NCD | `normalized_compression_distance(...)` |
