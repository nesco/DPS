"""This module contains all pixel-level operators on binary masks.
Binary masks are square grids, represented by lists of lists, of 0s (Nothing) and 1s (Something).

Operators are classified in:
    - Set operators: operators that only act at pixel level, without any interaction
    - Geometric operators: operators that involves grid information
    - Morphological operators: they involve notions of kernel
    - Topological operators: they involve notions of connectivity.
        Either 4-connectivity (Manhattan) or 8-connectivity (Chebyshev)
    - Signal and probabilistic operators: [TO-DO]
"""

from helpers import *

# connected_component
# - mask -> [][]
# - box -> ((top-left corner), (bottom_right, corner))
### Constants
# Topological constants
# The Manhattan / 4-connectivity kernel is a cross
kernel_manhattan = [[False, True, False], [True, True, True], [False, True, False]]
directions_manhattan = {'0': (-1, 0), '1': (0, -1), '2': (1, 0), '3': (0, 1)}

# The Chebyshev / 8-connectivity kernel is a square
kernel_chebyshev = [[True, True, True], [True, True, True], [True, True, True]]
directions_chebyshev = {
    "0": (-1, 0), "2": (0, -1), "4": (1, 0), "6": (0, 1),
    "1": (-1, -1), "3": (1, -1), "5": (1, 1), "7": (-1, 1)
}
### Operators on binary masks
## Helpers
def flatten(mask):
    height, width = proportions(mask)
    return [mask[row][col] for row in range(height) for col in range(width)]

def fill_unary(mask: Mask, condition: Callable[[bool], bool]) -> Mask:
    height, width = proportions(mask)
    mask_new = falses(height, width)
    for row in range(height):
        for col in range(width):
            mask_new[row][col] = condition(mask[row][col])
    return mask_new

def fill_binary(mask1: Mask, mask2: Mask, condition: Callable[[bool, bool], bool]) -> Mask:
    height, width = proportions(mask1)

    if proportions(mask2) != (height, width):
        msg = f"Applying binary filling to masks of incompatble dimensions: {(height, width)} vs {proportions(mask2)}"
        msg += f"\n mask n°1: {mask1}, mask n°2 {mask2}"
        raise ValueError(msg)

    mask_new = falses(height, width)
    for row in range(height):
        for col in range(width):
            if condition(mask1[row][col], mask2[row][col]):
                mask_new[row][col] = True
    return mask_new

## Set operators
# Set Mapping
def cardinal(mask: Mask) -> int:
    return sum(flatten(mask))

def intersection(mask1: Mask, mask2: Mask) -> Mask:
    """Intersection of two similar sized masks"""
    return fill_binary(mask1, mask2, lambda a, b: a and b)

def union(mask1: Mask, mask2: Mask) -> Mask:
    """Union of two similar sized canals"""
    return fill_binary(mask1, mask2, lambda a, b: a or b)

def complement(mask: Mask) -> Mask:
    return fill_unary(mask, lambda a: False if a else True)
# Set test
def set_contains(mask_larger, mask_smaller):
    return intersection(mask_larger, mask_smaller) == mask_smaller

def jaccard(mask1, mask2):
    canal_intersection = intersection(mask1, mask2)
    canal_union = union(mask1, mask2)

    cardinal_intersection = cardinal(canal_intersection)
    cardinal_union = cardinal(canal_union)

    if cardinal_union == 0:
        return 0
    else:
        return cardinal_intersection / cardinal_union

def jaccard_resized(mask1, mask2):
    # Determine the smaller dimensions
    height1, width1 = proportions(mask1)
    height2, width2 = proportions(mask2)
    max_height = max(height1, height2)
    max_width = max(width1, width2)

    # Create centered subgrids
    submask1 = create_centered_padded(mask1, max_height, max_width)
    submask2 = create_centered_padded(mask2, max_height, max_width)

    return jaccard(submask1, submask2)

## Geometric operators
def transpose(mask):
    height, width = proportions(mask)
    return [[mask[row][col] for row in range(height)] for col in range(width)]

def reverse_rows(mask):
    height, width = proportions(mask)
    return [[mask[row][col] for col in range(width)] for row in reversed(range(height))]

def reverse_cols(mask):
    height, width = proportions(mask)
    return [[mask[row][col] for col in reversed(range(width))] for row in range(height)]

def rotation(mask):
    return transpose(reverse_cols(mask))

def rotational_symmetries(mask, box):
    if mask == None or box == None:
        return 4

    submask = extract(mask, box)
    if submask == rotation(submask):
       return 4
    if submask == rotation(rotation(submask)):
       return 2
    return 0
## Morphological operators
def morphological_dilatation(mask_object: Mask, mask_kernel: Mask):
    """Returns the mask of the cropped morphological dilatation (~Minkowski addition) of an object and a structuring element """
    object_height, object_width = proportions(mask_object)
    kernel_height, kernel_width = proportions(mask_kernel)

    # The new mask will have the dimension of the object's mask
    # It's the only asymetry between "object" and "kernel"
    # Whereas traditional Minkowski addition treats them equally
    mask_new = falses(object_height, object_width)

    for row in range(object_height):
        for col in range(object_width):
            if mask_object[row][col]:
                for i in range(kernel_height):
                    for j in range(kernel_width):
                        k = row + i - kernel_height // 2
                        l = col + j - kernel_width // 2

                        if 0 <= k < object_height and 0 <= l < object_width:
                            if mask_kernel[i][j]:
                                mask_new[k][l] = True
    return mask_new


def morphological_erosion(mask_object: Mask, mask_kernel: Mask):
    """Returns the mask of the morphological erosion of an object using a structuring element."""
    object_height, object_width = proportions(mask_object)
    kernel_height, kernel_width = proportions(mask_kernel)

    # The new mask will have the dimension of the object's mask
    mask_new = falses(object_height, object_width)

    for row in range(object_height):
        for col in range(object_width):
            # Check if the kernel fits
            fits = True
            for i in range(kernel_height):
                for j in range(kernel_width):
                    k = row + i - kernel_height // 2
                    l = col + j - kernel_width // 2

                    if 0 <= k < object_height and 0 <= l < object_width:
                        if mask_kernel[i][j] and not mask_object[k][l]:
                            fits = False
                            break
                    else:
                        fits = False
                        break
                if not fits:
                    break

            if fits:
                mask_new[row][col] = True
            else:
                mask_new[row][col] = False

    return mask_new

def morphological_opening(mask_object: Mask, mask_kernel: Mask):
    return morphological_dilatation(morphological_erosion(mask_object, mask_kernel), mask_kernel)

def morphological_closing(mask_object: Mask, mask_kernel: Mask):
    return morphological_erosion(morphological_dilatation(mask_object, mask_kernel), mask_kernel)
## Topological operators
# Topological mappings
def dilatation(mask: Mask, chebyshev = True):
    if chebyshev:
        return morphological_dilatation(mask, kernel_chebyshev)
    return morphological_dilatation(mask, kernel_manhattan)

def erosion(mask: Mask, chebyshev = True):
    if chebyshev:
        return morphological_erosion(mask, kernel_chebyshev)
    return morphological_erosion(mask, kernel_manhattan)

def opening(mask, chebyshev = True):
    if chebyshev:
        return morphological_opening(mask, kernel_chebyshev)
    return morphological_opening(mask, kernel_manhattan)

def closing(mask, chebyshev = True):
    if chebyshev:
        return morphological_closing(mask, kernel_chebyshev)
    return morphological_closing(mask, kernel_manhattan)

# Topological tests
def is_exterior(mask_connected_component, chebyshev = True):
    """
       Determines if a connected component is exterior (touches the border).

       :param mask_connected_component: 2D list where 1 represents the component and 0 the background
       :param chebyshev: If True, use 8-connectivity; if False, use 4-connectivity
       :return: True if the component is exterior, False otherwise
    """

    kernel = kernel_chebyshev if chebyshev else kernel_manhattan
    height, width = proportions(mask_connected_component)
    mask = mask_connected_component

    def check_adjacent_cells(row, col):
        for i in range(3):
            for j in range(3):
                if kernel[i][j]:
                    adj_i, adj_j = row+i-1, col+j-1
                    if 0 <= adj_i < height and 0 <= adj_j < width:
                        if mask[adj_i][adj_j]:
                            return True

    # Check top and bottom borders
    for col in range(width):
        if check_adjacent_cells(0, col) or check_adjacent_cells(height-1, col):
                return True

    # Check left and right borders (excluding corners)
    for row in range(1, height-1):
        if check_adjacent_cells(row, 0) or check_adjacent_cells(row, width-1):
            return True

    return False

# Topological complex function
def connected_components(mask: Mask, chebyshev=False):
    """Extract connex components.
        Chebyshev: if True then 8-connexity is used instead of 4-connexity
    """
    component_number = 0
    height, width = proportions(mask)
    distribution = falses(height, width)
    to_see = [(i,j) for j in range(width) for i in range(height)]
    seen = set()
    components = {}

    for i,j in to_see:
        # Check if there is something to do
        if (i,j) in seen or not mask[i][j]:
            continue

        # New component found, increase the count and initiate queue
        component_number += 1
        components[component_number] = {'mask': [[0 for _ in range(width)] for _ in range(height)]}
        queue = [(i, j)]
        min_row, max_row = i, i
        min_col, max_col = j, j

        while queue:
            k, l = queue.pop(0)  # Correctly manage the queue
            if (k, l) in seen:
                continue

            distribution[k][l] = component_number
            components[component_number]['mask'][k][l] = 1
            seen.add((k, l))

            # Update the bounding box
            min_row, max_row = min(min_row, k), max(max_row, k)
            min_col, max_col = min(min_col, l), max(max_col, l)

            # Add all 4-connex neighbors
            for dk, dl in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nk, nl = k + dk, l + dl
                if 0 <= nk < height and 0 <= nl < width and mask[nk][nl] == 1 and (nk, nl) not in seen:
                    queue.append((nk, nl))

            if chebyshev:
                # Add diagonal neighbors
                for dk, dl in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nk, nl = k + dk, l + dl
                    if 0 <= nk < height and 0 <= nl < width and mask[nk][nl] == 1 and (nk, nl) not in seen:
                        queue.append((nk, nl))

        # Store the bounding box for the current component
        components[component_number]['box'] = ((min_row, min_col), (max_row, max_col))

    return component_number, distribution, components

def connected_components_coords(coords: Coords, props: Proportions, chebyshev=False):
    """Extract connex components.
        Chebyshev: if True then 8-connexity is used instead of 4-connexity
    """
    height, width = props
    component_number = 0
    to_see = coords.copy()
    seen = set()
    components = {}

    for i,j in to_see:
        # Check if there is something to do
        if (i,j) in seen or (i,j) not in coords:
            continue

        # New component found, increase the count and initiate queue
        component_number += 1
        components[component_number] = {'mask': set()}
        queue = [(i, j)]

        while queue:
            k, l = queue.pop(0)  # Correctly manage the queue
            if (k, l) in seen:
                continue

            components[component_number]['mask'].add((k,l))
            seen.add((k, l))

            # Add all 4-connex neighbors
            for dk, dl in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nk, nl = k + dk, l + dl
                if 0 <= nk < height and 0 <= nl < width and (nk,nl) in coords and (nk, nl) not in seen:
                    queue.append((nk, nl))

            if chebyshev:
                # Add diagonal neighbors
                for dk, dl in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nk, nl = k + dk, l + dl
                    if 0 <= nk < height and 0 <= nl < width and (nk, nl) in coords and (nk, nl) not in seen:
                        queue.append((nk, nl))


    return component_number, components


def partition(mask_connected_component, chebyshev=False):
    """Assuming the mask is one of a connected component of the grid / A single grid object
    it partition the grid into its "holes" and exterior regions, which are connected components
    of it's complement. An exterior regions is a region that has at least one 1 that is border adjacent.
    """
    mask_complement = complement(mask_connected_component)
    # Non-trivial: if the topology of the 1s follows 4-connectivity (manhattan)
    # then the topology of the 0s follows 8-connectivity (chebyshev)
    # and vice-versa
    component_number, distribution, components = connected_components(mask_complement, chebyshev=not chebyshev)
    components_partition = {}
    for key in components.keys():
        components_partition[key] = {'mask': components[key]['mask'],
            'is_exterior': is_exterior(components[key]['mask'])}

    return component_number, distribution, components_partition

def euler_characteristic(components_partition):
    """
    euler_characteristic compute the euler characteristic of a 2D object divided by 2.
    It does so using the "genus" of the object, i.e. the number of holes it has.
    :param components_partition: A dictionary of the connected components of the complement of the object.
    Augmented by the indication wether the comonent is a hole,
    or part of the exterior (it touches the border of the grid)
    """
    nb_holes = sum([0 if component['is_exterior'] == 1 else 1 for component in components_partition.values()])
    return 1 - nb_holes

def topological_contains(mask1, mask2, chebyshev=False):
    """
    Function used to determine if an object includes a other one, and the "size" of this inclusion,
    given by the number of dilatation to make it false.

    :return: None if mask1 doesn't contains mask2, otherwise it returns n >= 0,
    n being the number of dilatation during which mask1 still contains mask2
    """
    height, width = proportions(mask1)
    _, _, components_partition = partition(mask1, chebyshev=chebyshev)
    interior = mask1
    for key in components_partition.keys():
        if not components_partition[key]['is_exterior']:
            interior = union(interior, components_partition[key]['mask'])

    if not set_contains(interior, mask2):
        return None

    dilatation_number = 0
    mask_growing = dilatation(mask2, chebyshev=chebyshev)
    # Mask2 is dilated until it's no longer contained in the interior of mask1,
    # or if it filled the entire grid
    while set_contains(interior, mask_growing) and cardinal(mask_growing) < height*width:
        dilatation_number+=1
        mask_growing = dilatation(mask_growing, chebyshev=chebyshev)

    return dilatation_number

#def context(mask1, mask2, chebyshev = False):
#    kernel = kernel_chebyshev if chebyshev else kernel_manhattan

#### Functions of binary masks
## Tests
def _test_flatten():
    assert flatten([[1, 1], [0, 1]]) == [1, 1, 0, 1]

def _test_intersection():
    assert intersection([[1, 1],[0, 1]], [[1, 0], [1, 1]]) == [[1, 0], [0, 1]]
    assert intersection([[1, 0], [0, 1]], [[0, 1],[1, 0]]) == [[0, 0], [0, 0]]

def _test_union():
    assert union([[1, 0], [0, 1]], [[0, 1],[1, 0]]) == [[1, 1], [1, 1]]
    assert union([[1, 0], [0, 0]], [[0, 0], [0, 1]]) == [[1, 0], [0, 1]]

def _test_complement():
    assert complement([[1, 0], [0, 1]]) == [[0, 1], [1, 0]]

def _test_transpose():
    assert transpose([[1, 0], [0, 1]]) == [[1, 0], [0, 1]]
    assert transpose([[0, 1], [0, 0]]) == [[0, 0], [1, 0]]
    assert transpose([[0, 0], [1, 0]]) == [[0, 1], [0, 0]]

def _test_reverse_cols():
    assert reverse_cols([[1, 0], [1, 0]]) == [[0, 1], [0, 1]]
    assert reverse_cols([[0, 1], [0, 1]]) == [[1, 0], [1, 0]]

def _test_reverse_rows():
    assert reverse_rows([[1, 1], [0, 0]]) == [[0, 0], [1, 1]]
    assert reverse_rows([[0, 0], [1, 1]]) == [[1, 1], [0, 0]]
