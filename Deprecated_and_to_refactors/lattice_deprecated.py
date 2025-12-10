"""
The lattice class is an abstract representation of the Grid.
To make up  for the few data and compute, the lattice acts as a layer of abstraction upon the hard coded notion of topological connection.
It works because connected components seems to be a strong prior in the ARC challenge dataset.
"""

import traceback

from freeman import *
from grid import CoordsOperations, GridOperations, MaskOperations
from helpers import *
from loader import train_task_to_grids
from operators import *
from syntax_tree_copy import *


def extract_masks_bicolors(grid):
    """
    Returns a dict of all masks comprised of union of two colours
    """
    width, height = proportions(grid)
    colors_unique = list(
        set([grid[row][col] for row in range(height) for col in range(width)])
    )
    # colors_unique = list(grid_to_color_coords(grid))
    # masks_colors = split_by_color(grid)
    color_coords = grid_to_color_coords(grid)

    masks_bicolors = {}
    for i in range(len(colors_unique)):
        color_i = colors_unique[i]
        for j in range(i):
            color_j = colors_unique[j]
            masks_bicolors[(color_i, color_j)] = (
                color_coords[color_i] | color_coords[color_j]
            )

        masks_bicolors[(color_i, color_i)] = color_coords[color_i]

    return masks_bicolors


def list_components(
    mask_dict: dict[Any, Coords], proportions: Proportions, chebyshev=True
):
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
    components_seen = list()

    for key in mask_dict:
        coords = mask_dict[key]
        component_number, components = connected_components_coords(
            coords, proportions, chebyshev=chebyshev
        )

        for component in components.values():
            if component not in components_seen:
                components_seen.append(component)
    return components_seen


def rank_components(component_list):
    mask_list = [component["mask"] for component in component_list]
    size = len(mask_list)
    map_contains = MaskOperations.falses(size, size)
    depth_map = {}

    # First a tabular representation of the "contains" relation
    # on the connected components is created
    for row in range(size):
        for col in range(size):
            map_contains[row][col] = (
                mask_list[col] <= mask_list[row]
            )  # set_contains(mask_list[row], mask_list[col])

    # Then connected components are ranked by the number of other components they are in
    map_contained = transpose(map_contains)
    # Rising sea-level algorithm for depth
    depth_level = 1
    depth_map = {}
    current_depth = [
        indice for indice in range(size) if sum(map_contained[indice]) == 1
    ]

    # While there is some nodes at current depth
    while current_depth:
        depth_map[depth_level] = current_depth
        # removing the current node from map contained:
        for row in range(size):
            for col in current_depth:
                map_contained[row][col] = False

        depth_level += 1
        current_depth = [
            indice for indice in range(size) if sum(map_contained[indice]) == 1
        ]

    return depth_map, map_contains


def component_to_object(grid: ColorGrid, component, chebyshev=True):
    color_set = set([grid[row][col] for col, row in component["mask"]])
    component["colors"] = color_set


def construct_lattice_ends(grid: Grid):
    width, height = proportions(grid)
    mask_supremum = set([(col, row) for row in range(height) for col in range(width)])
    mask_infinmum = set()

    box_supremum = ((0, 0), (width - 1, height - 1))
    # colors_supremum = mask_colors(grid, mask_supremum)

    component_supremum = {
        "mask": mask_supremum,
        "box": box_supremum,
    }  # , 'euler':None, 'colors': colors_supremum}
    component_infimum = {
        "mask": mask_infinmum,
        "box": None,
    }  # , 'euler':1, 'colors': None}

    return component_supremum, component_infimum


def get_potential_starting_points(coordinates: list[Coord]) -> list[Coord]:
    if not coordinates:
        return []

    # Extract x and y coordinates
    col_coords, row_coords = zip(*coordinates)

    # Find the box boundaries:
    col_max, col_min = max(col_coords), min(col_coords)
    row_max, row_min = max(row_coords), min(row_coords)

    starts = []

    # 1. Adding corner points if they are in the mask
    corners = [
        (col_min, row_min),
        (col_max, row_min),
        (col_min, row_max),
        (col_max, row_max),
    ]

    starts.extend([corner for corner in corners if corner in coordinates])

    # 2. Edge points
    edges = (
        [(col, row_min) for col in range(col_min, col_max + 1)]  # Top edge
        + [(col, row_max) for col in range(col_min, col_max + 1)]  # Bottom edge
        + [(col_min, row) for row in range(row_min + 1, row_max)]  # Left edge
        + [(col_max, row) for row in range(row_min + 1, row_max)]  # Right edge
    )
    starts.extend(edge for edge in edges if edge in coordinates)

    # 3. The closest point to the approximate centroid
    col_centroid = sum(col_coords) // len(col_coords)
    row_centroid = sum(row_coords) // len(row_coords)

    distance_to_centroid = (
        lambda coord: (coord[0] - col_centroid) ** 2 + (coord[1] - row_centroid) ** 2
    )
    closest_to_centroid = min(coordinates, key=distance_to_centroid)
    starts.append(closest_to_centroid)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(starts))


def mask_to_ast(
    mask: Coords, colors: Colors, proportions: Proportions
) -> Optional[ASTNode]:
    width, height = proportions
    is_valid = lambda x: valid_coordinates(height, width, mask, x)
    start, ast = None, None
    starts = get_potential_starting_points(list(mask))

    def argmin_by_len_and_depth(lst):
        def key_func(node):
            return len(node) + 3 * get_depth(node)

        return min(range(len(lst)), key=lambda i: key_func(lst[i]))

    def get_traditional_construct():
        # Trying several combinations of start pos and dfs/bfs
        # choose the best option by iterating overstart points and dfs/bfs
        asts = []
        for start in starts:
            for method in ["dfs"]:  # ["dfs", "bfs"]:
                ast = Root(
                    start,
                    node["value"]["colors"],
                    construct_node1(start, is_valid, method),
                )
                asts.append(ast)
        return asts

    def get_freeman_construct() -> list[ASTNode]:
        freeman_nodes = []
        asts = []
        for start in starts:
            for method in TraversalModes:
                freeman_node = encode_connected_component(start, is_valid, method)
                freeman_nodes.append((start, freeman_node))

        for start, freeman_node in freeman_nodes:
            ast = Root(start, colors, construct_node(freeman_node))
            asts.append(ast)

        return asts

    if len(starts) >= 1:
        asts = get_freeman_construct()
        # Getting the smallest, regularized by the depth to avoid pathological cases
        asts = sorted(asts, key=len)
        asts = asts[:3]
        i = argmin_by_len_and_depth(asts)  # symbolize(asts, []))
        ast = asts[i]

    return ast


class Lattice:
    def __init__(self, grid: ColorGrid, mask_dict: dict[Any, Coords]):
        self.grid = grid
        self.width, self.height = proportions(grid)
        self.area = self.height * self.width
        self.refs: SymbolTable = []
        self.union_refs: SymbolTable = []
        self.unions: list[Optional[UnionNode]] = []

        # Retrieve all the connected components, except the entire grid if it is in

        component_ls = list_components(mask_dict, proportions(grid))
        components = list(filter(lambda x: len(x["mask"]) < self.area, component_ls))

        # Adding the supremum and the infimum as fake component
        component_supremum, component_infimum = construct_lattice_ends(grid)
        # components += [component_supremum, component_infimum]
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
            self.nodes.append(
                {"value": component, "successors": [], "predecessors": []}
            )

        # TO-DO: supremum should be at depth 0, correct rank_component
        # self.supremum  = len(self.nodes)-2 # Should be depth = 1
        # self.infimum = len(self.nodes)-1 # Should be depth = self.depth
        if component_supremum in self.nodes:
            self.supremum = self.nodes.index(component_supremum)
        else:
            self.supremum = len(self.nodes) - 1
            if len(self.depth_to_indices) > 1:
                self.nodes[self.supremum]["successors"] = self.depth_to_indices[2]

        if len(self.depth_to_indices) > 1:
            for i in self.depth_to_indices[2]:
                self.nodes[i]["predecessors"] = [self.supremum]

        self.infimum = -1

        def complete_lattice(successors, depth):
            if depth not in self.depth_to_indices:
                return None

            indices = self.depth_to_indices[depth]
            for indice in successors:
                # children are nodes of next rank included in it
                successors_indice = [i for i in indices if self.map_contains[indice][i]]
                self.nodes[indice]["successors"] = successors_indice
                for i in successors_indice:
                    self.nodes[i]["predecessors"].append(indice)
                complete_lattice(successors_indice, depth + 1)

            return successors

        if len(self.depth_to_indices) > 1:
            complete_lattice(self.nodes[self.supremum]["successors"], 3)

        def argmin_by_len_and_depth(lst):
            def key_func(node):
                return len(node) + 3 * get_depth(node)

            return min(range(len(lst)), key=lambda i: key_func(lst[i]))

        # Fetch codes for each nodes and add the current index at the same time
        for i, node in enumerate(self.nodes):
            node["index"] = i
            mask_list = node["value"]["mask"]
            ast = mask_to_ast(
                mask_list, node["value"]["colors"], (self.width, self.height)
            )

            node["value"]["ast"] = ast

            self.codes.append(ast)
        self.unions = []
        self.update_unions()

    def symbolize(self):
        self.codes = symbolize(self.codes, self.refs)

    def update_unions1(self):
        # A priority queue will be needed to track backgrounds. A simple (code, depth) would suffice at first
        to_process = set()
        processed = set()

        # First get the union for all simple unicolors nodes
        # Plus the placeholder for the final list

        self.unions = [None for _ in range(len(self.nodes))]
        for i, node in enumerate(self.nodes):
            if len(node["value"]["colors"]) == 1:
                to_process.add(i)

        # Then process all nodes
        while to_process:
            i = to_process.pop()
            box = CoordsOperations.box(self.nodes[i]["value"]["mask"])
            if len(self.nodes[i]["value"]["colors"]) == 1:
                self.unions[i] = construct_union(
                    None, [(i, self.codes[i])], self.refs, box
                )
            else:
                codes = []
                for j in self.nodes[i]["successors"]:
                    for k in self.unions[j].subindices:
                        codes.append((k, self.codes[k]))

                self.unions[i] = construct_union(self.codes[i], codes, self.refs, box)
                # self.unions[i] = construct_union([code for j in self.nodes[i]['successors'] for code in self.unions[j].codes], self.refs)

            processed.add(i)
            # Add predecessors that have all their successors processed
            for parent_i in self.nodes[i]["predecessors"]:
                if all(
                    child in processed for child in self.nodes[parent_i]["successors"]
                ):
                    to_process.add(parent_i)

        for i, node in enumerate(self.nodes):
            node_val = node["value"]
            shift_box = (
                -CoordsOperations.box(node_val["mask"])[0][0],
                -CoordsOperations.box(node_val["mask"])[0][1],
            )
            self.unions[i] = shift_ast(shift_box, self.unions[i])

        for i, node in enumerate(self.nodes):
            node_val = node["value"]["ast"] = self.unions[i]

        # The update

        # chain_code = serialize(node['value']['mask'])
        # start, code = extract_serialized_elements(chain_code)
        # morphology, coordinates = code_compression(code)
        # if morphology in self.morphologies:
        #    self.morphologies[morphology] += [(i, coordinates)]
        # else:
        #    self.morphologies[morphology] =  [(i, coordinates)]

        # if code in self.programs:
        #    self.programs[code] += [i]
        # else:
        #   self.programs[code] = [i]
        # node['value']['program'] = code
        # node['value']['start'] = start

    def update_unions(self):
        # A priority queue will be needed to track backgrounds. A simple (code, depth) would suffice at first
        to_process = set()
        processed = set()

        # First get the union for all simple unicolors nodes
        # Plus the placeholder for the final list

        self.unions = [None for _ in range(len(self.nodes))]
        for i, node in enumerate(self.nodes):
            if len(node["value"]["colors"]) == 1:
                to_process.add(i)

        # Then process all nodes
        while to_process:
            i = to_process.pop()
            box = CoordsOperations.box(self.nodes[i]["value"]["mask"])
            if len(self.nodes[i]["value"]["colors"]) == 1:
                self.unions[i] = construct_union(
                    None, [self.codes[i]], [], self.refs, box
                )
            else:
                codes = []
                unions = []
                for j in self.nodes[i]["successors"]:
                    if self.unions[j].background:
                        unions.append(self.unions[j])
                    else:
                        for k in self.unions[j].codes:
                            codes.append(k)

                self.unions[i] = construct_union(
                    self.codes[i], codes, unions, self.refs, box
                )
                if self.unions[i] is None:
                    print("self.unions[i] n°{i} is {self.unions[i]}")
                    print("Called construct_union with")
                    print(
                        f"self.codes[i]: {self.codes[i]}, codes={codes}, unions={unions}"
                    )
                # self.unions[i] = construct_union([code for j in self.nodes[i]['successors'] for code in self.unions[j].codes], self.refs)

            processed.add(i)
            # Add predecessors that have all their successors processed
            for parent_i in self.nodes[i]["predecessors"]:
                if all(
                    child in processed for child in self.nodes[parent_i]["successors"]
                ):
                    to_process.add(parent_i)

        for i, node in enumerate(self.nodes):
            node_val = node["value"]
            shift_box = (
                -CoordsOperations.box(node_val["mask"])[0][0],
                -CoordsOperations.box(node_val["mask"])[0][1],
            )
            self.unions[i] = shift_ast(shift_box, self.unions[i])

        for i, node in enumerate(self.nodes):
            node_val = node["value"]["ast"] = self.unions[i]

        # The update

        # chain_code = serialize(node['value']['mask'])
        # start, code = extract_serialized_elements(chain_code)
        # morphology, coordinates = code_compression(code)
        # if morphology in self.morphologies:
        #    self.morphologies[morphology] += [(i, coordinates)]
        # else:
        #    self.morphologies[morphology] =  [(i, coordinates)]

        # if code in self.programs:
        #    self.programs[code] += [i]
        # else:
        #   self.programs[code] = [i]
        # node['value']['program'] = code
        # node['value']['start'] = start


def display_unions(lattice: Lattice):
    for i, union in enumerate(lattice.unions):
        try:
            original_code = lattice.codes[i]
            unsymb_original = unsymbolize(original_code, lattice.refs)
            print(f"Node {i}:")
            print(f"  Original code: {original_code}")
            print(f"  Length: {len(original_code)}")
            print(f"  Unsymbolized: {unsymb_original}")
            print(f"  Union: {union}")
            print("  Subcodes:")
            if union is None:
                return
            for j, code in enumerate(union.codes, 1):
                unsymb_code = unsymbolize([code], lattice.refs)[0]
                print(f"    {j}. {code}")
                print(f"       Length: {len(code)}")
                print(f"       Unsymbolized: {unsymb_code}")
            print()  # Empty line for readability between nodes
        except Exception as e:
            print(f"Error processing node {i}: {str(e)}")
            print("".join(traceback.format_tb(e.__traceback__)))


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


def symbolize_together(lattices):
    # for l in lattices:
    #    l.symbolize()

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)

    for i, l in enumerate(lattices):
        l.refs = nrefs
        l.codes = update_asts(l.codes, nrefs, mappings[i])

    # List of list of ASTs to list of ASTs
    lcodes = [code for l in lattices for code in l.codes]
    lcodes, nrefs = symbolize(lcodes, nrefs)

    # Reconstruction of lattices codes
    index = 0
    for i, l in enumerate(lattices):
        for j in range(len(l.codes)):
            l.codes[j] = lcodes[index]
            index += 1
        l.refs = nrefs
        l.update_unions()

    lucodes = []
    urefs = []
    for l in lattices:
        print("New lattice ----")
        for m, u in enumerate(l.unions):
            print(f"Union: {u} n°{m}")
            lucodes.append(u.background)
            for code in u.codes:
                print(f"with {code}")
                lucodes.append(code)

    lucodes, urefs = symbolize(lucodes, urefs)

    # Reconstruction of union codes
    index = 0
    for i, l in enumerate(lattices):
        for j, u in enumerate(l.unions):
            u.background = lucodes[index]
            nucodes = set()
            index += 1
            for k in range(len(u.codes)):
                nucodes.add(lucodes[index])
                index += 1
            u.codes = nucodes
        l.union_refs = urefs


def symbolize_together1(lattices):
    # for l in lattices:
    #    l.symbolize()

    refs_ls = [l.refs for l in lattices]
    nrefs, mappings = fuse_refs(refs_ls)

    for i, l in enumerate(lattices):
        l.refs = nrefs
        l.codes = update_asts(l.codes, nrefs, mappings[i])

    # List of list of ASTs to list of ASTs
    lcodes = [code for l in lattices for code in l.codes]
    lcodes = symbolize(lcodes, nrefs)

    # Reconstruction of lattices codes
    index = 0
    for i, l in enumerate(lattices):
        for j in range(len(l.codes)):
            l.codes[j] = lcodes[index]
            index += 1
        l.update_unions()

    lucodes = []
    urefs = []
    for l in lattices:
        for u in l.unions:
            lucodes.append(u.background)
            for code in quotient_to_set(u.codes):
                lucodes.append(code)

    lucodes = symbolize(lucodes, urefs)

    # Reconstruction of union codes
    index = 0
    for i, l in enumerate(lattices):
        for j, u in enumerate(l.unions):
            u.background = lucodes[index]
            nucodes = set()
            index += 1
            for k in range(len(u.codes)):
                nucodes.add(lucodes[index])
                index += 1
            u.codes = set_to_quotient(lambda x: x.colors, nucodes)
        l.union_refs = urefs


def distance1(node_val1, node_val2, refs) -> tuple[float, Coord]:
    prog1 = unsymbolize(node_val1["ast"], refs)
    prog2 = unsymbolize(node_val2["ast"], refs)

    _, _, points1 = decode(prog1)
    _, _, points2 = decode(prog2)

    return distance_jaccard_optimal(points1, points2)


def distance(node_val1, node_val2, refs) -> tuple[float, Coord]:
    prog1 = unsymbolize(node_val1["ast"], refs)
    prog2 = unsymbolize(node_val2["ast"], refs)

    _, points1 = decode(prog1)
    _, points2 = decode(prog2)

    return distance_jaccard_optimal(points1, points2)


def align_lattice(lattice1, lattice2):
    correspondances = []
    bottom1 = lattice1.infimum["predecessors"]
    bottom2 = lattice2.infimum["predecessors"]

    for ib1 in bottom1:
        for ib2 in bottom2:
            obj1 = lattice1.nodes[ib1]["value"]
            obj2 = lattice2.nodes[ib2]["value"]

            submask1 = extract(obj1["mask"], obj1["box"])
            submask2 = extract(obj2["mask"], obj2["box"])

            color_common = obj1["colors"] & obj2["colors"]
            color_union = obj1["colors"] | obj2["colors"]

            # Color distance, topological distance, metric distance
            jaccard_color = 1 - len(color_common) / len(color_union)
            jaccard_card = 1 - jaccard_resized(submask1, submask2)
            card_max = max(cardinal(submask1), cardinal(submask2))
            card_min = min(cardinal(submask1), cardinal(submask2))
            card_diff = (card_max - card_min) / card_max
            # distance_euler = abs(obj1['euler'] - obj2['euler']) / max(abs(obj1['euler']), abs(obj2['euler']))

            # correspondances.append(((ib1, ib2), jaccard_color + distance_euler + jaccard_card + card_diff))

    return None  # correspondances


def test_align(inputs):
    masks0 = extract_masks_bicolors(inputs[0])
    masks1 = extract_masks_bicolors(inputs[1])
    lattice0 = Lattice(inputs[0], masks0)
    lattice1 = Lattice(inputs[1], masks1)
    # corr = align_lattice(lattice0, lattice1)
    corr = None
    return corr, lattice0, lattice1


def input_to_lattice(input: ColorGrid) -> Lattice:
    masks = extract_masks_bicolors(input)
    return Lattice(input, masks)


# correspondances, lattice1, lattice2 = test_align(inputs)
# corr = sorted(correspondances, key=lambda el: el[1])


def union_to_grid1(union, refs, height=None, width=None):
    unsymb_union = unsymbolize(union, refs)
    _, _, points = decode(unsymb_union)
    if height is None or width is None:
        grid = GridOperations.from_points(points)
        GridOperations.print(grid)
    else:
        grid = zeros(height, width)
        GridOperations.populate(grid, points)
    return grid


def union_to_grid(union, refs, height=None, width=None):
    unsymb_union = unsymbolize(union, refs)
    _, points = decode(unsymb_union)
    if height is None or width is None:
        grid = GridOperations.from_points(points)
        GridOperations.print(grid)
    else:
        grid = zeros(height, width)
        GridOperations.populate(grid, points)
    return grid


def lattice_to_grid(lattice: Lattice) -> Grid:
    lattice.update_unions()
    unsymb_union = unsymbolize(lattice.unions[lattice.supremum], lattice.refs)
    if unsymb_union is None:
        raise ValueError("Supremum union is None")
    # populate(grid, unsymb_union)
    return node_to_grid(unsymb_union)


def shifted(shift, union, refs):
    unsymb = unsymbolize(union, refs)
    if unsymb is None:
        return None
    return factor_by_refs(shift_ast(shift, unsymbolize(union, refs)), refs)


def print_symb():
    tasks = ["2dc579da.json", "48d8fb45.json"]
    l = []
    for task in tasks:
        inputs, outputs, _, _ = train_task_to_grids(task)
        l_inputs = [input_to_lattice(input) for input in inputs]
        l_outputs = [input_to_lattice(output) for output in outputs]
        l += l_inputs + l_outputs

    symbolize_together(l)
    print("Symbol Table")
    for i, ref in enumerate(l[0].refs):
        print(f"Symbol n°{i}: {ref}")
        print("Linked programs:")
        for la in l:
            for code in la.codes:
                if i in get_symbols(code):
                    print(f"Code: {code}")
                    print(f"Unsymbolized: {unsymbolize(code, l[0].refs)}")
        print("\n")


def problem2(task="2dc579da.json"):
    inputs, outputs, _, _ = train_task_to_grids(task)
    l_inputs = [input_to_lattice(input) for input in inputs]
    l_outputs = [input_to_lattice(output) for output in outputs]
    l = l_inputs + l_outputs
    symbolize_together(l)
    print("Symbol Table")
    for ref in l[0].refs:
        print(f"{ref}")

    print("Optimal mappings:")
    for i, out in enumerate(l_outputs):
        union_out = out.unions[out.supremum]
        node_out = out.nodes[out.supremum]
        refs = l_inputs[i].union_refs

        print(f"For problem {i}")

        # Add unshifted unions
        dist = [
            (j, distance(inp["value"], node_out["value"], refs))
            for j, inp in enumerate(l_inputs[i].nodes)
        ]
        dist1 = [
            (j, ast_distance(inp, union_out, refs))
            for j, inp in enumerate(l_inputs[i].unions)
        ]
        # for j, inp in enumerate(l_inputs[i].nodes):
        #    node_val = inp['value']
        # shift is - top-left corner
        #    shift = -CoordsOperations.box(node_val['mask'])[0][0], -CoordsOperations.box(node_val['mask'])[0][1]
        #    dist.append((j, distance(shifted(shift, node_val, refs), node_out['value'], refs), True))

        dist_min = min(dist, key=lambda x: x[1][0])
        dist_min1 = min(dist1, key=lambda x: x[1])
        union_min = l_inputs[i].unions[dist_min[0]]
        union_min1 = l_inputs[i].unions[dist_min1[0]]

        # Shift the best union to it's proper frmae?'
        node_val_min = l_inputs[i].nodes[dist_min[0]]["value"]
        # shift = -node_val_min['box'][0][0], -node_val_min['box'][0][1]

        # shift = factor_by_refs(shift_ast(shift, unsymbolize([union_min], refs)[0]), refs)
        # shift_box = -CoordsOperations.box(node_val_min['mask'])[0][0], -CoordsOperations.box(node_val_min['mask'])[0][1]
        shift_proper = dist_min[1][1][0], dist_min[1][1][1]
        print("Target:")
        print(f"Union Out: {union_out}, len: {len(union_out)}")
        print(f"Unsymoblized: {unsymbolize(union_out, refs)}")
        union_to_grid(union_out, refs)
        print(f"Using Jaccard / Pixel Space:")
        print(f"Distance: {dist_min[1][0]}")
        print(
            f"Distance: {distance_jaccard(decode(unsymbolize(union_min, refs))[2], decode(unsymbolize(union_out, refs))[2])}"
        )
        print(f"Program distance: {ast_distance(union_min, union_out, refs)}")
        print(f"Unshifted input: {union_min} -> {union_out}")
        print(f"Input:")
        union_to_grid(union_min, refs)
        print(f"Shift : {shift_proper}")
        print(f"index {dist_min[0]}")

        print(f"Using Levenhstein / Latent Space")
        union_to_grid(union_min1, refs)
        print(f"Distance Program: {dist_min1[1]}")
        print(f"Unshifted input: {union_min1} -> {union_out}")
        # print(f'Shift : {shift_proper}')
        print(f"index {dist_min1[0]}")


def problem1(task="2dc579da.json"):
    inputs, outputs, _, _ = train_task_to_grids(task)
    l_inputs = [input_to_lattice(input) for input in inputs]
    l_outputs = [input_to_lattice(output) for output in outputs]
    l = l_inputs + l_outputs
    symbolize_together(l)
    print("Symbol Table")
    for i, ref in enumerate(l[0].union_refs):
        print(f"Symbol n°{i}: {ref}")

    print("Optimal mappings:")
    for i, out in enumerate(l_outputs):
        union_out = out.unions[out.supremum]
        node_out = out.nodes[out.supremum]
        refs = l[i].union_refs  # l_inputs[i].union_refs

        # Add unshifted unions

        dist = [
            (j, distance(inp["value"], node_out["value"], refs))
            for j, inp in enumerate(l_inputs[i].nodes)
        ]
        dist1 = [
            (j, ast_distance(inp, union_out, refs))
            for j, inp in enumerate(l_inputs[i].unions)
        ]
        # for j, inp in enumerate(l_inputs[i].nodes):
        #    node_val = inp['value']
        # shift is - top-left corner
        #    shift = -CoordsOperations.box(node_val['mask'])[0][0], -CoordsOperations.box(node_val['mask'])[0][1]
        #    dist.append((j, distance(shifted(shift, node_val, refs), node_out['value'], refs), True))

        dist_min = min(dist, key=lambda x: x[1][0])
        # dist_min1 = min(dist1, key=lambda x: x[1])
        dist_sort1 = sorted(dist1, key=lambda x: x[1])
        dist_min1 = dist_sort1[0]
        union_min = l_inputs[i].unions[dist_min[0]]
        union_min1 = l_inputs[i].unions[dist_min1[0]]

        # Shift the best union to it's proper frmae?'
        node_val_min = l_inputs[i].nodes[dist_min[0]]["value"]
        # shift = -node_val_min['box'][0][0], -node_val_min['box'][0][1]

        # shift = factor_by_refs(shift_ast(shift, unsymbolize([union_min], refs)[0]), refs)
        # shift_box = -CoordsOperations.box(node_val_min['mask'])[0][0], -CoordsOperations.box(node_val_min['mask'])[0][1]
        shift_proper = dist_min[1][1][0], dist_min[1][1][1]

        print(f"For problem {i}")

        print("Target:")
        print(f"Union Out: {union_out}, len: {len(union_out)}")
        union_to_grid(union_out, refs)
        print(f"Using Jaccard / Pixel Space:")
        print(f"Distance: {dist_min[1][0]}")
        print(
            f"Distance: {distance_jaccard(decode(unsymbolize(union_min, refs))[2], decode(unsymbolize(union_out, refs))[2])}"
        )
        print(f"Program distance: {ast_distance(union_min, union_out, refs)}")
        print(f"Unshifted input: {union_min} -> {union_out}")
        print(f"Input:")
        union_to_grid(union_min, refs)
        print(f"Shift : {shift_proper}")
        print(f"index {dist_min[0]}")

        print(f"Using Levenhstein / Latent Space")
        union_to_grid(union_min1, refs)
        print(f"Distance Program: {dist_min1[1]}")
        print(f"Unshifted input: {union_min1} -> {union_out}")
        print(f"Unsymoblized: {unsymbolize(union_min1, refs)}")

        # print(f'Shift : {shift_proper}')
        print(f"index {dist_min1[0]}")


def problem(task="2dc579da.json"):
    inputs, outputs, _, _ = train_task_to_grids(task)
    l_inputs = [input_to_lattice(input) for input in inputs]
    l_outputs = [input_to_lattice(output) for output in outputs]
    l = l_inputs + l_outputs
    symbolize_together(l)
    print("Symbol Table")
    for i, ref in enumerate(l[0].union_refs):
        print(f"Symbol n°{i}: {ref}")

    print("Optimal mappings:")
    for i, out in enumerate(l_outputs):
        union_out = out.unions[out.supremum]
        node_out = out.nodes[out.supremum]
        refs = l[i].union_refs  # l_inputs[i].union_refs

        # Add unshifted unions

        dist = [
            (j, distance(inp["value"], node_out["value"], refs))
            for j, inp in enumerate(l_inputs[i].nodes)
        ]
        dist1 = [
            (j, ast_distance(inp, union_out, refs))
            for j, inp in enumerate(l_inputs[i].unions)
        ]
        # for j, inp in enumerate(l_inputs[i].nodes):
        #    node_val = inp['value']
        # shift is - top-left corner
        #    shift = -CoordsOperations.box(node_val['mask'])[0][0], -CoordsOperations.box(node_val['mask'])[0][1]
        #    dist.append((j, distance(shifted(shift, node_val, refs), node_out['value'], refs), True))

        dist_min = min(dist, key=lambda x: x[1][0])
        # dist_min1 = min(dist1, key=lambda x: x[1])
        dist_sort1 = sorted(dist1, key=lambda x: x[1])
        dist_min1 = dist_sort1[0]
        union_min = l_inputs[i].unions[dist_min[0]]
        union_min1 = l_inputs[i].unions[dist_min1[0]]

        # Shift the best union to it's proper frmae?'
        node_val_min = l_inputs[i].nodes[dist_min[0]]["value"]
        # shift = -node_val_min['box'][0][0], -node_val_min['box'][0][1]

        # shift = factor_by_refs(shift_ast(shift, unsymbolize([union_min], refs)[0]), refs)
        # shift_box = -CoordsOperations.box(node_val_min['mask'])[0][0], -CoordsOperations.box(node_val_min['mask'])[0][1]
        shift_proper = dist_min[1][1][0], dist_min[1][1][1]

        print(f"For problem {i}")

        print("Target:")
        print(f"Union Out: {union_out}, len: {len(union_out)}")
        print(f"Unsymbolized: {unsymbolize(union_out, refs)}")
        union_to_grid(union_out, refs)
        print(f"Using Jaccard / Pixel Space:")
        print(f"Distance: {dist_min[1][0]}")
        print(
            f"Distance: {distance_jaccard(decode(unsymbolize(union_min, refs))[1], decode(unsymbolize(union_out, refs))[1])}"
        )
        print(f"Program distance: {ast_distance(union_min, union_out, refs)}")
        print(f"Unshifted input: {union_min} -> {union_out}")
        print(f"Input:")
        union_to_grid(union_min, refs)
        print(f"Shift : {shift_proper}")
        print(f"index {dist_min[0]}")

        print(f"Using Levenhstein / Latent Space")
        union_to_grid(union_min1, refs)
        print(f"Distance Program: {dist_min1[1]}")
        print(f"Unshifted input: {union_min1} -> {union_out}")
        print(f"Unsymoblized: {unsymbolize(union_min1, refs)}")

        # print(f'Shift : {shift_proper}')
        print(f"index {dist_min1[0]}")


def solve_problem(task="2dc579da.json"):
    inputs, outputs, input_test, output_test = load_final(task)

    l_inputs = [input_to_lattice(input) for input in inputs]
    l_outputs = [input_to_lattice(output) for output in outputs]
    l_input_test = input_to_lattice(input_test)
    l_output_test = input_to_lattice(output_test)

    l = l_inputs + l_outputs + [l_input_test]
    symbolize_together(l)
    print("Symbol Table")
    for i, ref in enumerate(l[0].union_refs):
        print(f"Symbol n°{i}: {ref}")

    print("Optimal mappings:")

    input_space = []
    output_space = []
    pair_spaces = []

    for i, out in enumerate(l_outputs):
        union_out = out.unions[out.supremum]
        node_out = out.nodes[out.supremum]
        refs = l[i].union_refs  # l_inputs[i].union_refs

        output_space.append(union_out)

        # Add unshifted unions

        dist = [
            (j, ast_distance(inp, union_out, refs))
            for j, inp in enumerate(l_inputs[i].unions)
        ]

        # dist_min1 = min(dist1, key=lambda x: x[1])
        dist_sort = sorted(dist, key=lambda x: x[1])
        dist_min = dist_sort[0]
        union_min = l_inputs[i].unions[dist_min[0]]

        # Shift the best union to it's proper frmae?'
        node_val_min = l_inputs[i].nodes[dist_min[0]]["value"]
        # shift = -node_val_min['box'][0][0], -node_val_min['box'][0][1]

        shift_proper = (
            -CoordsOperations.box(node_val_min["mask"])[0][0],
            -CoordsOperations.box(node_val_min["mask"])[0][1],
        )
        # shift_proper = dist_min[1][1][0], dist_min[1][1][1]

        print(f"For problem {i}")

        print("Target:")
        print(f"Union Out: {union_out}, len: {len(union_out)}")
        union_to_grid(union_out, refs)
        print(f"Program distance: {ast_distance(union_min, union_out, refs)}")
        print(f"Unshifted input: {union_min} -> {union_out}")
        print(f"Input:")
        union_to_grid(union_min, refs)
        print(f"Shift : {shift_proper}")
        print(f"index {dist_min[0]}")
        print(f"Unsymoblized: {unsymbolize(union_min, refs)}")
        input_space.append(union_min)

        pair_spaces.append([union_min, union_out])

        # print(f'Shift : {shift_proper}')
    print("Predicted test comp:")
    min_dist = None
    min_object = None
    refs = l_input_test.union_refs

    dists = []
    for i, union in enumerate(l_input_test.unions):
        # Compute the distance to the input space as the sum of distances of it's objects
        # What we want might more be akin to a projection, so the distance to the min of the distances. Test it
        dist = sum([ast_distance(union, inp, refs) for inp in input_space])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_object = union
        dists.append((dist, union))

    print(f"best objs: {min_object}")
    union_to_grid(min_object, refs)
    inputs = [min_object] + input_space
    outputs = []

    # distance = lambda a, b: ast_distance(a, b, refs)
    def distance(a, b):
        match a, b:
            case SymbolicNode(i1, _, _), SymbolicNode(i2, _, _) if i1 == i2:
                return 0
            case _, _:
                return ast_distance(a, b, refs)

    def asts_print(ast_set):
        for ast in ast_set:
            print(ast)

    def ast_set_print(ast_set):
        for i, ast in ast_set:
            print(f"Element n°{i} : {ast}")

    def list_of_ast_set_print(list_ast_set):
        for i, ast_set in enumerate(list_ast_set):
            print(f"Group n°{i}")
            ast_set_print(ast_set)

    category_input = set_to_category(
        [input.codes | {input.background} for input in inputs if input],
        distance,
    )
    # invariants_in, cliques_in, morphisms_in  = set_to_category([input.codes | {input.background} for input in inputs if input], distance)
    # print(f'Invariants In: ')
    # ast_set_print(invariants_in)

    # print(f'Cliques In')
    # list_of_ast_set_print(cliques_in)

    # print(f'Morphisms In: {morphisms_in}')
    # dists.sort(key= lambda x: x[0], reverse=True)
    category_output = set_to_category(
        [output.codes | {output.background} for output in output_space if output],
        distance,
    )
    # invariants_out, cliques_out, morphisms_out  = set_to_category([output.codes | {output.background} for output in output_space if output], distance)
    # print(f'Invariants Out:')
    # ast_set_print(invariants_out)

    # print(f'Cliques Out:')
    # list_of_ast_set_print(cliques_out)

    # print(f'Morphisms Out: {morphisms_out}')

    pair_categories = []

    for i, pair in enumerate(pair_spaces):
        # print(f'\n Pair: {i}')
        pair_category = set_to_category(
            [u.codes | {u.background} for u in pair if u], distance
        )
        # invariants, cliques, morphisms  = comma_category.invariants, comma_category.cliques, comma_category.morphisms
        pair_categories.append(pair_category)
        # print(f'Invariants:')
        # asts_print(invariants)

        # print(f'Cliques :')
        # list_of_ast_set_print(cliques)
    natural_transformations = functor_categories_to_natural_transformation_category(
        pair_categories, category_input, category_output
    )
    transform = natural_transformation_category_to_callable(natural_transformations)

    input_enumeration = {
        (category_input.element_to_class_index(code), code)
        for code in min_object.codes
        if code
    }
    input_background_index = category_input.element_to_class_index(
        min_object.background
    )

    output_codes = set()

    for input_index, input_code in input_enumeration:
        if input_index is not None and transform(input_index) is not None:
            output_codes.add(transform(input_index)(input_code))

    output_background_index = transform(input_background_index)(min_object.background)

    # output_codes = {output_indices}
    output = UnionNode(codes=output_codes, background=output_background_index)

    print(natural_transformations)
    GridOperations.print(union_to_grid(output, refs))

    # print('rest')
    # for dist, u in dists:
    #    print(f'Object: {u} has dsit: {dist}')
    #    union_to_grid(u, refs)


def natural_transformation_category_to_callable(
    natural_transformation_category,
):
    @optional
    def transformation(index: IndexElement):
        for functor in natural_transformation_category.invariants:
            input_index, output_index, type = functor
            if index == input_index:
                if type == INVARIANT:
                    return IDENTITY
                else:
                    return None
        return None

    return transformation


# element_to_equivalence_class_index#def output_index_to_code(index, )
