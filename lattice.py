"""
The lattice class is an abstract representation of the Grid.
To make up  for the few data and compute, the lattice acts as a layer of abstraction upon the hard coded notion of topological connection.
It works because connected components seems to be a strong prior in the ARC challenge dataset.
"""

from helpers import *
from operators import *
from syntax_tree import *


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

def list_components(mask_dict, chebyshev = False):
    """
    list_components creates a list of unique connected components extracted
    from a dictionnary of masks.
    The resulting list of connected components should all have distinct masks over the grid.

    :param dict mask_dict: The dictionnary of masks from which to extract connected components
    chebyshev : boolean
        Wether to use the Chebyshev topology (8-connectivity/L_{inf}),
        instead of the manhattan topology (4-connectivity/L_{1})

    Returns
    -------
    component_list : list
        The list of unique connected components extracted from `mask_dict`
    """
    component_list = []
    for key in mask_dict:
        mask = mask_dict[key]
        masks_components_seen = [component['mask'] for component in component_list]
        component_number, distribution, components = connected_components(mask, chebyshev=chebyshev)
        component_list += [component for component in components.values()
            if component['mask'] not in masks_components_seen]
    return component_list

def rank_components2(component_list):
    mask_list = [component['mask'] for component in component_list]
    size = len(mask_list)
    map_contains = zeros(size, size)
    rank_map = {}

    # First a tabular representation of the "contains" relation
    # on the connected components is created
    for row in range(size):
        for col in range(size):
            map_contains[row][col] = set_contains(mask_list[row], mask_list[col])

    # Then connected components are ranked by the number of other components they are in
    map_contained = transpose(map_contains)
    for i in range(size):
        # number of objects containing it, the diagonal is included because it serves a proxy
        # that the object is contained by the grid
        rank = sum(map_contained[i])
        if rank not in rank_map.keys():
            rank_map[rank] = [i]
        else:
            rank_map[rank].append(i)


    return rank_map, map_contains

def rank_components(component_list):
    mask_list = [component['mask'] for component in component_list]
    size = len(mask_list)
    map_contains = zeros(size, size)
    depth_map = {}

    # First a tabular representation of the "contains" relation
    # on the connected components is created
    for row in range(size):
        for col in range(size):
            map_contains[row][col] = set_contains(mask_list[row], mask_list[col])

    # Then connected components are ranked by the number of other components they are in
    map_contained = transpose(map_contains)
    # Rising sea-level algorithm for depth
    depth_level = 1
    depth_map = {}
    current_depth = [indice for indice in range(size) if sum(map_contained[indice]) == 1]

    # While there is some nodes at current depth
    while current_depth:
        depth_map[depth_level] = current_depth
        # removing the current node from map contained:
        for row in range(size):
            for col in current_depth:
                map_contained[row][col] = 0

        depth_level += 1
        current_depth = [indice for indice in range(size) if sum(map_contained[indice]) == 1]


    return depth_map, map_contains

def component_to_object(grid, component, chebyshev = False):
    color_set = mask_colors(grid, component['mask'])
    _, _, component_partition = partition(component['mask'], chebyshev=chebyshev)
    euler = euler_characteristic(component_partition)
    rsymmetries = rotational_symmetries(component['mask'], component['box'])
    component['colors'] = color_set
    component['euler'] = euler
    component['rsymmetries'] = rsymmetries

def construct_object_lattice(grid, chebyshev=False): #deprecated
    height, width = proportions(grid)
    masks_bicolors = extract_masks_bicolors(grid)
    #components = {'manhattan': None, 'chebyshev': None}
    # objects = {mask, boundaries, children}

    components = list_components(masks_bicolors, chebyshev=chebyshev)
    # Remove a grid object if it already exist for bi colored grid:
    components = [component for component in components if cardinal(component['mask']) < height*width]

    depth_map, map_contains = rank_components(components)

    # Remove a grid object if it already exist for bi colored grid:
    grid_mask = ones(height, width)
    grid_box = ((0, 0), (height-1, width-1))
    grid_object = {'mask': grid_mask, 'box': grid_box}

    nodes = []
    for i, component in enumerate(components):
        component_to_object(grid, component)
        nodes.append({'value': component, 'successors': [], 'predecessors': []})

    supremum = {'value': grid_object, 'successors': [], 'predecessors': []}
    supremum['successors'] = [(indice, nodes[indice]) for indice in depth_map[1]]
    grid_object['colors'] = mask_colors(grid, grid_mask)
    #grid_object['euler'] = sum([node['value']['euler'] for _, node in supremum['successors']
    #    if node['value'] is not None])


    def complete_tree(children, depth):
        if depth not in depth_map:
            return None

        indices = depth_map[depth]
        for indice, node in children:
            # children are nodes of next rank included in it
            children_indice = [(i, nodes[i]) for i in indices if map_contains[indice][i]]
            node['successors'] = children_indice
            for _, node_successor in children_indice:
                node_successor['predecessors'] += node
            complete_tree(children_indice, depth+1)

        return children
    complete_tree(supremum['successors'], 2)

    def clean_of_indice(tree):
        if tree['successors'] is not None:
            for _, child in tree['successors']:
                clean_of_indice(child)
            tree['successors'] = [child for _, child in tree['successors']]


    clean_of_indice(supremum)
    return supremum

def construct_lattice_ends(grid):
    height, width = proportions(grid)
    mask_supremum = ones(height, width)
    mask_infinmum = zeros(height, width)

    box_supremum = ((0, 0), (height-1, width-1))
    #colors_supremum = mask_colors(grid, mask_supremum)

    component_supremum = {'mask': mask_supremum, 'box': box_supremum}#, 'euler':None, 'colors': colors_supremum}
    component_infimum = {'mask': mask_infinmum, 'box': None}#, 'euler':1, 'colors': None}

    return component_supremum, component_infimum

def get_potential_starting_points(coordinates: List[Coord]) -> List[Coord]:
    if not coordinates:
        return []

    # Extract x and y coordinates
    x_coords, y_coords = zip(*coordinates)

    # Find the box boundaries:
    x_max, x_min = max(x_coords), min(x_coords)
    y_max, y_min = max(y_coords), min(y_coords)

    starts = []

    # 1. Adding corner points if they are in the mask
    corners = [
        (x_min, y_min), (x_max, y_min),
        (x_min, y_max), (x_max, y_max)
    ]

    starts.extend([corner for corner in corners if corner in coordinates])

     # 2. Edge points
    edges = (
        [(x, y_min) for x in range(x_min, x_max + 1)] +  # Bottom edge
        [(x, x_max) for x in range(x_min, x_max + 1)] +  # Top edge
        [(x_min, y) for y in range(y_min + 1, y_max)] +  # Left edge
        [(x_max, y) for y in range(y_min + 1, y_max)]    # Right edge
    )
    starts.extend(edge for edge in edges if edge in coordinates)

    # 3. The closest point to the approximate centroid
    x_centroid = sum(x_coords) // len(x_coords)
    y_centroid = sum(y_coords) // len(y_coords)

    distance_to_centroid = lambda coord: (coord[0] - x_centroid)**2 + (coord[1] - y_centroid)**2
    closest_to_centroid =min(coordinates, key=distance_to_centroid)
    starts.append(closest_to_centroid)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(starts))

class Lattice():

    def __init__(self, grid, mask_dict):
        self.grid = grid
        self.height, self.width = proportions(grid)
        self.area = self.height * self.width
        self.refs = []

        # Retrieve all the connected components, except the entire grid if it is in
        component_ls = list_components(mask_dict)
        components = list(filter(lambda x: cardinal(x['mask']) < self.area, component_ls))

        # Adding the supremum and the infimum as fake component
        component_supremum, component_infimum = construct_lattice_ends(grid)
        #components += [component_supremum, component_infimum]
        components += [component_supremum]

        self.depth_to_indices, self.map_contains = rank_components(components)
        self.depth = max(self.depth_to_indices)
        self.indice_to_depth = {}
        for depth, indices in self.depth_to_indices.items():
            for indice in indices:
                self.indice_to_depth[indice] = depth


        self.nodes = []
        self.codes = []

        for component in components:
            component_to_object(grid, component)
            self.nodes.append({'value': component, 'successors': [], 'predecessors': []})

        #TO-DO: supremum should be at depth 0, correct rank_component
        #self.supremum  = len(self.nodes)-2 # Should be depth = 1
        #self.infimum = len(self.nodes)-1 # Should be depth = self.depth
        self.supremum  = len(self.nodes)-1
        self.infimum = -1

        self.nodes[self.supremum]['successors'] = self.depth_to_indices[2]
        for i in self.depth_to_indices[2]:
            self.nodes[i]['predecessors'] = [self.supremum]

        def complete_lattice(successors, depth):
            if depth not in self.depth_to_indices:
                return None

            indices = self.depth_to_indices[depth]
            for indice in successors:
                # children are nodes of next rank included in it
                successors_indice = [i for i in indices if self.map_contains[indice][i]]
                self.nodes[indice]['successors'] = successors_indice
                for i in successors_indice:
                    self.nodes[i]['predecessors'].append(indice)
                complete_lattice(successors_indice, depth+1)

            return successors

        complete_lattice(self.nodes[self.supremum]['successors'], 3)

        def argmin_by_len_and_depth(lst):
            def key_func(node):
                return len(node) + 3*get_depth(node)
            return min(range(len(lst)), key=lambda i: key_func(lst[i]))

        # Fetch codes for each nodes and add the current index at the same time
        for i, node in enumerate(self.nodes):

            node['index'] = i
            mask_list = grid_to_list(node['value']['mask'])
            is_valid = lambda x: valid_coordinates(self.height, self.width, set(mask_list), x)

            start, ast = None, None
            starts = get_potential_starting_points(mask_list)
            if len(starts)>=1:
                # Trying several combinations of start pos and dfs/bfs
                # choose the best option by iterating overstart points and dfs/bfs
                asts=[]
                for start in starts:
                    for method in ["dfs", "bfs"]:
                        ast = Root(start, node['value']['colors'], construct_node(start, is_valid, method))
                        asts.append(ast)
                        #asts.append(ast_map(factorize_moves, ast))

                # Getting the smallest, regularized by the depth to avoid pathological cases
                asts = sorted(asts, key=len)
                asts = asts[:3]
                i = argmin_by_len_and_depth(asts)#symbolize(asts, []))
                ast = asts[i]

                #ast = Root(start, node['value']['colors'], construct_node(start, is_valid, "dfs"))

            node['value']['ast'] = ast

            self.codes.append(ast)
        self.unions = []
        self.update_unions()

    def symbolize(self):
        self.codes = symbolize(self.codes, self.refs)
    def update_unions(self):
        # A priority queue will be needed to track backgrounds. A simple (code, depth) would suffice at first
        to_process = set()
        processed = set()

        # First get the union for all simple unicolors nodes
        # Plus the placeholder for the final list

        self.unions = [None for _ in range(len(self.nodes))]
        for i, node in enumerate(self.nodes):
            if len(node['value']['colors']) == 1:
                to_process.add(i)

        # Then process all nodes
        while to_process:
            i = to_process.pop()
            if len(self.nodes[i]['value']['colors']) == 1:
                self.unions[i] = construct_union([self.codes[i]], self.refs)
            else:
                self.unions[i] = construct_union([code for j in self.nodes[i]['successors'] for code in self.unions[j].codes], self.refs)
            processed.add(i)
            # Add predecessors that have all their successors processed
            for parent_i in self.nodes[i]['predecessors']:
                if all(child in processed for child in self.nodes[parent_i]['successors']):
                    to_process.add(parent_i)



        # The update

            #chain_code = serialize(node['value']['mask'])
            #start, code = extract_serialized_elements(chain_code)
            #morphology, coordinates = code_compression(code)
            #if morphology in self.morphologies:
            #    self.morphologies[morphology] += [(i, coordinates)]
            #else:
            #    self.morphologies[morphology] =  [(i, coordinates)]

            #if code in self.programs:
            #    self.programs[code] += [i]
            #else:
            #   self.programs[code] = [i]
            #node['value']['program'] = code
            #node['value']['start'] = start

def display_unions(lattice: Lattice):
    for node, union in lattice.unions.items():
        try:
            original_code = lattice.codes[node]
            unsymb_original = unsymbolize([original_code], lattice.refs)[0]
            print(f"Node {node}:")
            print(f"  Original code: {original_code}")
            print(f"  Length: {len(original_code)}")
            print(f"  Unsymbolized: {unsymb_original}")
            print(f"  Union: {union}")
            print("  Subcodes:")
            for i, code in enumerate(union.codes, 1):
                unsymb_code = unsymbolize([code], lattice.refs)[0]
                print(f"    {i}. {code}")
                print(f"       Length: {len(code)}")
                print(f"       Unsymbolized: {unsymb_code}")
            print()  # Empty line for readability between nodes
        except Exception as e:
            print(f"Error processing node {node}: {str(e)}")

def align_lattice2(lattices):

    for l in lattices:
        l.symbolize()

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)
    for i, l in enumerate(lattices):
        l.refs = nrefs
        l.codes = update_asts(l.codes, nrefs, mappings[i])

    ref_weight = {}
    ref_weight_lattices = []
    for l in lattices:
        ref_weight_lattices.append({})
        for code in l.codes:
            symbols = get_symbols(code)
            for s in symbols:
                if s in ref_weight:
                    ref_weight[s] += 1
                else:
                    ref_weight[s] = 1
                if s in ref_weight_lattices[-1]:
                    ref_weight_lattices[-1][s] += 1
                else:
                    ref_weight_lattices[-1][s] = 1

    return ref_weight, ref_weight_lattices

def align_lattice1(lattices):

    #for l in lattices:
    #    l.symbolize()

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)

    for i, l in enumerate(lattices):
        l.refs = nrefs
        l.codes = update_asts(l.codes, nrefs, mappings[i])

    print(f"Nref: {nrefs}")

    # List of list of ASTs to list of ASTs
    lcodes = [code for l in lattices for code in l.codes]
    lcodes = symbolize(lcodes, nrefs)

    # Reconstruction of lattices codes
    index = 0
    for i, l in enumerate(lattices):
        for j in range(len(l.codes)):
            l.codes[j] = lcodes[index]
            index += 1

    ref_weight = {}
    ref_weight_lattices = []
    for l in lattices:
        ref_weight_lattices.append({})
        for code in l.codes:
            symbols = get_symbols(code)
            for s in symbols:
                if s in ref_weight:
                    ref_weight[s] += 1
                else:
                    ref_weight[s] = 1
                if s in ref_weight_lattices[-1]:
                    ref_weight_lattices[-1][s] += 1
                else:
                    ref_weight_lattices[-1][s] = 1

    return ref_weight, ref_weight_lattices

def align_lattice(lattice1, lattice2):
    correspondances = []
    bottom1 = lattice1.infimum['predecessors']
    bottom2 = lattice2.infimum['predecessors']

    for ib1 in bottom1:
        for ib2 in bottom2:
            obj1 = lattice1.nodes[ib1]['value']
            obj2 = lattice2.nodes[ib2]['value']

            submask1 = extract(obj1['mask'], obj1['box'])
            submask2 = extract(obj2['mask'], obj2['box'])

            color_common = obj1['colors'] & obj2['colors']
            color_union = obj1['colors'] | obj2['colors']

            # Color distance, topological distance, metric distance
            jaccard_color = 1 - len(color_common) / len(color_union)
            jaccard_card = 1 - jaccard_resized(submask1, submask2)
            card_max = max(cardinal(submask1), cardinal(submask2))
            card_min = min(cardinal(submask1), cardinal(submask2))
            card_diff = (card_max - card_min)/card_max
            #distance_euler = abs(obj1['euler'] - obj2['euler']) / max(abs(obj1['euler']), abs(obj2['euler']))

            #correspondances.append(((ib1, ib2), jaccard_color + distance_euler + jaccard_card + card_diff))

    return None#correspondances
def test_align(inputs):
    masks0 = extract_masks_bicolors(inputs[0])
    masks1 = extract_masks_bicolors(inputs[1])
    lattice0 = Lattice(inputs[0], masks0)
    lattice1 = Lattice(inputs[1], masks1)
    #corr = align_lattice(lattice0, lattice1)
    corr = None
    return corr, lattice0, lattice1
def input_to_lattice(input):
    masks = extract_masks_bicolors(input)
    return Lattice(input, masks)
# correspondances, lattice1, lattice2 = test_align(inputs)
# corr = sorted(correspondances, key=lambda el: el[1])

def lattice_to_grid_old(lattice):
    grid = zeros(lattice.height, lattice.width)
    unsymb = unsymbolize(lattice.codes, lattice.refs)
    for code in unsymb:
        if isinstance(code, Root) and len(code.colors)==1:
            populate(grid, code)
    return grid

def lattice_to_grid(lattice):
    grid = zeros(lattice.height, lattice.width)
    lattice.update_unions()
    unsymb_union = unsymbolize([lattice.unions[lattice.supremum]], lattice.refs)[0]
    populate(grid, unsymb_union)
    return grid

def coordinating(input_ls, ouput_ls):
    inputs, outputs, tree = load()
    l_inputs = [input_to_lattice(input) for input in inputs]
    l_outputs = [input_to_lattice(output) for output in outputs]
    l = l_inputs+l_outputs
