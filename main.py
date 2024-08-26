## Imports

import math
from os import CLD_CONTINUED
from helpers import *
from operators import *
from lattice import *
# Tree primitives
# String:
    # value: move string
    # children: None
# Repeat:
    # value : 2 <= N < 33 (5bits)
    # children : [STR]
#

#### Object as programs created by DFS


def construct_tree(code):
    return {'type': 'string', 'value': code, 'children':[]}

class Object():
    def __init__(self, mask, box, colors, euler,  rsymmetrie):
        self.mask = mask
        self.box = box
        self.colors = colors
        self.euler = euler
        self.rsymmetrie = rsymmetrie

## Functions
# Helpers

# structure
# - train
#   - []
#       - input [[]]
#       - output [[]]
# - test
#    - [] (len 1)
#      - input [[]]
#      - output [[]]

def filter_by_color(grid, color):
    height, width = proportions(grid)
    grid_new = zeros(height, width)

    for i in range(height):
        for j in range(width):
            if grid[i][j] == color:
                grid_new[i][j] = 1

    return grid_new



def split_by_color(grid):
    """Create for each color a mask, i.e a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids

def extract_masks_bicolors(grid):
    """
    Returns a dict of all masks comprised of union of two colours
    """
    colors_unique = list(colors_extract(grid))
    masks_colors = split_by_color(grid)
    masks_bicolors = {}
    for i in range(len(colors_unique)):
        color_i = colors_unique[i]
        for j in range(i):
            color_j = colors_unique[j]
            masks_bicolors[(color_i, color_j)] = union(masks_colors[color_i], masks_colors[color_j])

        masks_bicolors[(color_i, color_i)] = masks_colors[color_i]
    return masks_bicolors

# Operator over masks:

# inputs, outputs, tree = load()
# correspondances, lattice1, lattice2 = test_align(inputs)
# mask = lattice2.nodes[6]['value']['mask']
# chain_code = serialize(mask)
# grid_new = zeros(*proportions(mask))
# unserialize(grid_new, chain_code)

def test_freeman():
    inputs, outputs, tree = load()
    correspondances, lattice1, lattice2 = test_align(inputs)
    mask = lattice2.nodes[6]['value']['mask']
    chain_code = serialize(mask)
    grid_new = zeros(*proportions(mask))
    unserialize(grid_new, chain_code)
    return mask, chain_code, grid_new

#mask, chain_code, grid_new = test_freeman()
# Morphological operators

# Signal-theory operator

def cross_correlation(mask1, mask2):
    # Determine which grid is larger and which is the kernel
    if len(mask1) * len(mask1[0]) >= len(mask2) * len(mask2[0]):
        large = mask1
        kernel = mask2
    else:
        large = mask2
        kernel = mask1

    large_rows, large_cols = len(large), len(large[0])
    kernel_rows, kernel_cols = len(kernel), len(kernel[0])

    # Dimensions of the correlation matrix
    corr_rows = large_rows - kernel_rows + 1
    corr_cols = large_cols - kernel_cols + 1

    corr_matrix = [[0. for _ in range(corr_rows)] for _ in range(corr_cols)]

    for i in range(corr_rows):
        for j in range(corr_cols):
            sum_prod = 0
            for k in range(kernel_rows):
                for l in range(kernel_cols):
                    # Corresponding coordinates in the large grid
                    y = i + k
                    x = j + l
                    sum_prod += large[x][y] * kernel[k][l]

            sum_kernel = sum([kernel[i][j] for i in range(kernel_rows) for j in range(kernel_cols)])
            #sum_large = sum([large[i][j] for i in range(large_rows) for j in range(large_cols)])
            if  sum_kernel == 0:
                corr_matrix[i][j] = 0
            else:
                corr_matrix[i][j] = sum_prod / sum_kernel

    return corr_matrix


# Set distances and similarities
def jaccard1(mask1, mask2):
    """Jaccard index for two canals of same grid size"""
    rows, cols = len(mask1), len(mask1[0])
    total = rows*cols
    count11 = 0.
    count00 = 0.

    for i in range(rows):
        for j in range(cols):
            if mask1[i][j] == 1 and mask2[i][j] == 1:
                count11 += 1.
            if mask1[i][j] == 0 and mask2[i][j] == 0:
                count00 += 1.

    if count00 == total:
        return 0., -1, -1

    index = count11 / (total - count00)
    return index, -1, -1


def inter_over_union(mask1, mask2):
    canal_intersection = intersection(mask1, mask2)
    canal_union = union(mask1, mask2)

    cardinal_intersection = cardinal(canal_intersection)
    cardinal_union = cardinal(canal_union)

    if cardinal_union == 0:
        return 0., -1, -1
    else:
        return cardinal_intersection / cardinal_union, -1, -1

def correlation_peak(mask1, mask2):
    correlation_matrix = cross_correlation(mask1, mask2)
    max = -1
    max_i, max_j = -1, -1
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix[0])):
            if correlation_matrix[i][j] > max:
                max = correlation_matrix[i][j]
                max_i, max_j = i,j

    return max, max_i, max_j

def canal_similarity_matrix(similarity, grid1, grid2):
    canals1 = split_by_color(grid1)
    canals2 = split_by_color(grid2)

    similarity_matrix = [[0. for _ in range(10)] for _ in range(10)]

    for i in range(10):
       for j in range(10):
            similarity_matrix[i][j], _, _ = similarity(canals1[i], canals2[j])

    return similarity_matrix
# Signal distance and similarities
## Operators
# Position absolute:
    # Intersection
    # Intersection with pooling
# Position relative
    # Convolution
    # Convolution with pooling
def load_task(task, index):
    data = read_path('training/' + task)
    grid_prob = data['train'][index]['input']
    grid_sol = data['train'][index]['output']
    grids_prob = split_by_color(grid_prob)
    grids_sol = split_by_color(grid_sol)

    return data, grid_prob, grid_sol, grids_prob, grids_sol

# data, grid_prob, grid_sol, grids_prob, grids_sol = load_task(task)
# geometric notion: jaccard
# topologic notion: connex component + correlation


# TO-DO list:
    # Spin number for connected mask
    # inclusion trees for connected masks + top image as root with # of dilatatio
    # euler's number for connected masks
    # density number for connected masks (basically (card(mask) - 2) ) / (card(bounding box full) - 2))
    # where diamond is object of smallest cardinal ccupying a given bounding box
    # Basically two pixels indicating the corners

### Type

def print_lattice(grid, lattice):
    if lattice is not None:
        print_colored_grid(extract(grid, lattice['value']['box']))
        print_colored_grid(lattice['value']['mask'])

    if lattice['successors'] is not None:
        for el in lattice['successors']:
            print_lattice(grid, el)

def load(task = "2dc579da.json"):
    data = read_path('training/' + task)
    inputs = [el['input'] for el in data['train']]
    outputs = [el['output'] for el in data['train']]
    tree = construct_object_lattice(inputs[0])
    return inputs, outputs, tree
# inputs, outputs, tree = load()

def get_sizes():
    data, uuids = get_all()
    category = []
    for task in data:
        inputs = [el['input'] for el in task['train']]
        outputs = [el['output'] for el in task['train']]

        # Som are both in pairwise and common ouputs, see which is best to begin with
        # test for common size across ouputs
        output_prop = proportions(outputs[0])
        common_output = True
        for i in range(1, len(outputs)):
            if proportions(outputs[i]) != output_prop:
                common_output = False
                break

        if common_output:
            category.append(f"Common Output")#: {output_prop}")
            continue

        # Testing for pairwise size
        prop_out = [proportions(output) for output in outputs]
        prop_in =  [proportions(input) for input in inputs]
        common_pair = (prop_out == prop_in)
        #common_pair = True
        #for i, input in enumerate(inputs):

           # if proportions(outputs[i]) != proportions(input):
            #    common_pair = False
            #    break

        if common_pair:
            category.append(f"Common Pair")
            continue

        # testing for components
        in_components = True
        l_propins = []
        for input in inputs:
            masks = extract_masks_bicolors(input)
            component_ls = list_components(masks)
            l_propins.append([prop_box(comp['box']) for comp in component_ls])

        l_propouts = [proportions(output) for output in outputs]

        for i, propout in enumerate(l_propouts):
            if not propout in l_propins[i]:
                in_components = False

        if in_components:
            category.append(f"In components")
            continue


        category.append('else')
    return category, uuids

def get_counts():
    category, _ = get_sizes()
    counts = {}
    for cat in category:
        if cat in counts:
            counts[cat] += 1
        else:
            counts[cat] = 1
    return counts

def get_else():
    category, uuids = get_sizes()
    uu = []
    for i, cat in enumerate(category):
        if cat == "else":
            uu.append(uuids[i])
    return uu

def test_compress():
    inputs, outputs, tree = load()
    correspondances, lattice1, lattice2 = test_align(inputs)
    progs = [node['value']['program'] for node in lattice2.nodes]
    comp = [code_compression(code) for code in progs]
    return progs, comp

def test_symbolize():
    inputs, outputs, tree = load()
    lattices = [input_to_lattice(input) for input in inputs]
    error = False
    codes = []

    for i, l in enumerate(lattices):
        codes.append(l.codes)
        l.symbolize()
        symb = l.codes
        unsymb = unsymbolize(symb, l.refs)
        for j, c in enumerate(codes[i]):
            if c != unsymb[j]:
                print("\n")
                print("Error during single lattice symbolization")
                print(f"Lattice n°{i}")
                print(f"Code n°{j}")
                print(f"Original code: {c}")
                print(f"Symbolized code: {symb[j]}")
                print(f"Unsymbolized code: {unsymb[j]}")
                error = True

    if error:
        print("Errors during single lattice alignement")

    print("################")
    print("################")

    lattices = [input_to_lattice(input) for input in inputs]
    align_lattice2(lattices)
    if not ((lattices[0].refs == lattices[1].refs)  and (lattices[0].refs == lattices[2].refs)):
        print("Symbol tables alignement failed")

    refs = lattices[0].refs
    for i, l in enumerate(lattices):
        symb = l.codes
        unsymb = unsymbolize(symb, refs)
        for j, c in enumerate(codes[i]):
            if c != unsymb[j]:
                print("\n")
                print("Error during multi lattice symbolization")
                print(f"Lattice n°{i}")
                print(f"Code n°{j}")
                print(f"Original code: {c}")
                print(f"Symbolized code: {symb[j]}")
                print(f"Unsymbolized code: {unsymb[j]}")
                error = True

def test_simplify_repetitions():
    test_cases = [
        "230",      # Should return itself
        "22300322",  # Alternating pattern
        "111222333",  # Simple repetition
        "123123123",  # Repeating sequence
        "12345",     # No repetition
        "1212121",   # Alternating
        "11111",     # Single character repetition
        "112233112233",  # Multiple repetitions
        "1",         # Single move
        "11",        # Two identical moves
        "12",        # Two different moves
    ]

    for case in test_cases:
        moves = Moves(case)
        simplified = moves.simplify_repetitions()
        print(f"Original: {moves}, {moves.__repr__()}")
        print(f"Simplified: {moves}, {simplified.__repr__()}")
        print(f"Type: {type(simplified).__name__}")
        print(f"Is original object: {moves == simplified}")
        print()

def test_fuse_refs():
    inputs, outputs, tree = load()
    lattices = [input_to_lattice(input) for input in inputs]
    # constructing refs
    for l in lattices:
        l.symbolize()

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)
    errors = 0

    for i, refs in enumerate(refs_ls):
        print(f"Symbols of lattice n°{i}")
        for j, ref in enumerate(refs):
            print(f"Reference n°{j} == {ref}")
            print(f"Mapped to reference n°{mappings[i][j]} == {nrefs[mappings[i][j]]}")
            print(f"By the mapping {j} --> {mappings[i][j]}")
            print("\n")
            if nrefs[mappings[i][j]] != ref:
                errors += 1

    print(f"Number of errors detected: {errors}")

def test_factor_by_refs():
    inputs, outputs, tree = load()
    lattices = [input_to_lattice(input) for input in inputs]
    ccodes = [[code.copy() if isinstance(code, ASTNode) else code for code in l.codes] for l in lattices]
    errors = 0
    for l in lattices:
        l.symbolize()
    for i, lc in enumerate(ccodes):
        print(f"For lattice n°{i}")
        print(f"Symbol table:")
        for ref in lattices[i].refs:
            print(ref)
        print("\n")
        for j, code in enumerate(lc):
            code = factor_by_refs(code, lattices[i].refs)
            if lattices[i].codes[j] != code:
                print(f"Original factorized code: {lattices[i].codes[j]}, of lenght: {len(lattices[i].codes[j])}")
                print(f"Independently factorized code: {code}, of lenght: {len(code)}")
                errors += 1

        print("\n")
    print(f"Errors dectected: {errors}")

def test_functionalized():

    fun = functionalized(Branch([Moves('1'), Repeat(Moves('2'), 3), Moves('3'), Moves('4')]))
    res = [(Branch(sequences=[Moves(moves='1'), Variable(-1), \
        Moves(moves='3'), Moves(moves='4')]), Repeat(node=Moves(moves='2'), count=3))]

    print(f"Branch: {fun == res}")

    fun = functionalized(Root((0, 0), {4}, Moves('111')))
    res = [
        (Root(start=Variable(-1), colors={4}, node=Moves(moves='111')), (0, 0)),
        (Root(start=(0, 0), colors={4}, node=Variable(-1)), Moves(moves='111')),
        (Root(start=(0, 0), colors=Variable(-1), node=Moves(moves='111')), {4})
    ]
    print(f"Root: {fun == res}")

def test_update_asts():
    inputs, outputs, tree = load()
    lattices = [input_to_lattice(input) for input in inputs]
    codes = [l.codes for l in lattices]

    for l in lattices:
        l.symbolize()

    for i, l in enumerate(lattices):
        for j, c in enumerate(l.codes):
            if unsymbolize([c], l.refs)[0] != codes[i][j]:
                print(f"Issue during single lattice symbolization")
                print(f"For lattice n°{i}, code n°{j}: ")
                print(f"{c} is unsymbolized to:")
                print(f"{unsymbolize([c], l.refs)[0]} which is different from: ")
                print(f"{codes[i][j]}")

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)

    for i, refs in enumerate(refs_ls):
        for j, ref in enumerate(refs):
            if unsymbolize([ref], refs) !=  unsymbolize([nrefs[mappings[i][j]]], nrefs):
                print(f"Issue during fusing: ")
                print(f"For lattice n°{i}, reference n°{j} is")
                print(f"{ref} which was mapped to")
                print(f"{nrefs[mappings[i][j]]} through the mapping {j} -> {mappings[i][j]}")


    for i, l in enumerate(lattices):
        for j, c in enumerate(l.codes):
            ncode = update_asts([c], nrefs, mappings[i])[0]
            if unsymbolize([ncode], nrefs)[0] != codes[i][j]:
                print("Error during unsymbolization process after trying to update the ast with the new symbol table")
                print(f"For lattice n°{i}, code n°{j}")
                print(f"The failing update code is {ncode}")
                print(f"which resolves to {unsymbolize([ncode], nrefs)[0]}")
                print(f"And it doesn't get back to the original contrarily to the previous code: {c}")
                print(f"which resolves to {unsymbolize([c], l.refs)[0]}")

                #print("Differences :")
                #print(f"First code")

                print("Symbols: ")
                for k, ref in enumerate(l.refs):
                    print(f"ref n°{k} : {ref}")
                    print(f"nref n°{mappings[i][k]} : {nrefs[mappings[i][k]]}")

                #print("\n")

                #for node in c.breadth_iter():
                #    if isinstance(node, SymbolicNode):
                #        print(f"Symbolic node: {node}")
                #        print(f"With parameter: {node.param}")
                #        print(f"Which refers to: {l.refs[node.index]}")
                #        print(f"which resolves to: {resolve_symbolic(node, l.refs)}")
                #for node in ncode.breadth_iter():
                #    if isinstance(node, SymbolicNode):
                #        print(f"Symbolic node: {node}")
                #        print(f"With parameter: {node.param}")
                #        print(f"Which refers to: {nrefs[node.index]}")
                #        print(f"which resolves to: {resolve_symbolic(node, nrefs)}")

        #unsymb = unsymbolize(l.codes, l.refs)
        #nunsymb = unsymbolize(ncodes, nrefs)
        #if unsymb != nunsymb:
        #    for j, u in enumerate(unsymb):
        #        print(f"Code n°{j}")
        #        print(f"Original code: {codes[i][j]}")
        #        print(f"Unsymbolization with single lattice symbol table: {u}")
        #        print(f"Unsymbolization with multj lattices symbol table: {nunsymb[j]}")
        print("\n")
