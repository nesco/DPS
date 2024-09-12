"""
Rectangles are the closed sets of a 2D grid. In the same fashion masks where obtained
by conditioning the grid by sets of colors, they should be by rectangles.

They should respect the following criteria:
    * Dimensions n, m >= 1 and area >= 1. No criteria, points are rectangles

    * A shape cannot be comprised of two rectangles,
    one of which is entirely comprised in another
    * Thus the rectangles have to have the max size possible

    * There can be only up to 2 overlapping rectangles

Rectangles are parametrized from their top-left corner.
So they are similar to Box

Naming conventions:
    'cxxxx' -> 'current_xxxx' / 'xxxxx_current' or 'copy_xxxxxx' / 'xxxx_copy'
    'nxxxx' -> 'new_xxxxx' / 'xxxxx_new' or 'next_xxxx' / 'xxxxx_next'
    'pxxxx' -> 'previous_xxxx' / 'xxxxx_previous'
"""

from typing import Final
from helpers import *
from operators import condition_by_connected_components
from freeman import mask_to_boundary

DIMENSION_SIZE_REQUIREMENT: Final[int] = 1
AREA_REQUIREMENT: Final[int] = 1
MAX_OVERLAP: Final[int] = 1000 # No Max Overlap

I_MIN_COL = 0
I_MIN_ROW = 1
I_WIDTH = 2
I_HEIGHT = 3

Rectangle = tuple[int, int, int, int] # (start col, start row, width, height)
OrthogonalSet = set[Rectangle] # Set comprised of rectangles

def box_to_rectangle(box: Box) -> Rectangle:
    (col_min, row_min), (col_max, row_max) = box
    width, height = col_max - col_min + 1, row_max - row_min + 1
    return (col_min, row_min, width, height)

def rectangle_to_box(rect: Rectangle) -> Box:
    col_min, row_min, width, height = rect
    col_max, row_max = col_min + width - 1, row_min + height - 1
    return ((col_min, row_min), (col_max, row_max))

def is_rectangle_included(rect1: Rectangle, rect2: Rectangle) -> bool:
    box1 = rectangle_to_box(rect1)
    box2 = rectangle_to_box(rect2)
    return is_box_included(box1, box2)

def rectangle_to_area(rect: Rectangle) -> int:
    _, _, width, height = rect
    return width*height

def rectangle_in_coords(coords: Coords, rect: Rectangle) -> bool:
    coords_rect = set(iter_rect(rect))
    return coords_rect <= coords

def find_coverage_by_large_sub_rectangles(coords: Coords) -> list[Rectangle]:
    """
    We do not return the largest sub rectangle per say,
    but the largest one that ressembles the most to a square
    """
    # To find the largest rectangle,
    # first find the largest for all starting points in the boundary
    # Then return the one with the greatest area
    # And then start cherching for another rectangle by loking for in starting points not seen
    boundary = mask_to_boundary(coords)
    coverage = set()
    def is_in_coords(rect: Rectangle):
        return rectangle_in_coords(coords, rect)
    def find_largest_rectangle_by_starting_point(start: Coord) -> Optional[Rectangle]:
        # Expand the rectangle as much as possible both in width and height simultaneously.
        #To truly find the largest rectangle, you should compare it with the largest
        # obtained by expanding it by width first
        # and the largest by expanding it by height first
        start_col, start_row = start
        width = height = 1

        while True:
            # Expand the rectangle to the right:
            expanded = False
            if is_in_coords((start_col, start_row, width+1, height)):
                width += 1
                expanded = True
            # Expand the rectangle to the left
            if is_in_coords((start_col-1, start_row, width+1, height)):
                start_col -= 1
                width += 1
                expanded = True
            # Expand the rectangle downwards:
            if is_in_coords((start_col, start_row, width, height+1)):
                height += 1
                expanded = True
            # Expand the rectangle Upwards:
            if is_in_coords((start_col, start_row-1, width, height+1)):
                start_row -= 1
                height += 1
                expanded = True

            if not expanded:
                break

        if width == 0 or height == 0:
           return None
        return (start_col, start_row, width, height)

    # Get the largest recrangle by every border starting points
    rectangles: list[Rectangle] = []
    for start in boundary:
        rect = find_largest_rectangle_by_starting_point(start)
        if rect:
            rectangles.append(rect)

    # Order them by decreasing area
    rectangles.sort(key=rectangle_to_area, reverse=True)

    # Add them progressively if they add new points,
    # until the coords mask is completely covered
    nrectangles: list[Rectangle] = []
    for rect in rectangles:
        ccoverage = set(iter_rect(rect))
        if not ccoverage <= coverage:
            nrectangles.append(rect)
            coverage.update(ccoverage)

    # Because single points are valid rect here,
    # the entire mask should be covered
    if not coverage == coords:
        raise ValueError('The mask is not entirely covered by rectangles, something went wrong.')

    return nrectangles

def find_sub_rectangles(coords: Coords, overlapping=True) -> list[Rectangle]:
    """
    For each starting point, `find_sub_rectangles` will collect the largest rectangle there can be
    """
    if not coords:
        return list()

    cols, rows = unpack_coords(coords)
    col_min, col_max = min(cols), max(cols)
    row_min, row_max = min(rows), max(rows)

    rectangles: list[Rectangle] = list()
    seen: Coords = set()

    for start_col in range(col_min, col_max + 1):
        for start_row in range(row_min, row_max + 1):
            # Check if the top right corner belongs to the mask of the shape
            # Or if it belongs to an already processed area
            if not (start_col, start_row) in coords or (start_col, start_row) in seen:
                continue

            width = height = 0
            # Expand the rectangle to the right:
            while start_col + width <= col_max and (start_col + width, start_row) in coords:
                if not overlapping and (start_col + width, start_row) in seen:
                    break
                width += 1

            # Expand the rectangle downwards:
            while start_row + height <= row_max and all((start_col + dcol, start_row + height) in coords for dcol in range(width)):
                if not overlapping and any((start_col + dcol, start_row + height) in seen for dcol in range(width)):
                    break
                height += 1

            # Check if the rectangle meets the minimum size requirements:
            if width >= DIMENSION_SIZE_REQUIREMENT \
            and height >= DIMENSION_SIZE_REQUIREMENT \
            and width * height >= AREA_REQUIREMENT:
                    nrect = (start_col, start_row, width, height)
                    rectangles.append(nrect)
                    seen.update(iter_rect(nrect)) # Mark the area as seen

    return rectangles

def filter_and_order_rectangles_deprectaed(rectangles: list[Rectangle]) -> list[Rectangle]:
    # First order them by decreasing area
    # is_rectangle_included should not be useful anymore
    # As there is at least an edge not included
    rectangles.sort(key=rectangle_to_area, reverse=True)
    return [rect for i, rect in enumerate(rectangles)
               if not any(is_rectangle_included(rect, prect)
                          for prect in rectangles[:i])]

def iter_rect(rect: Rectangle) -> Iterator[Coord]:
    col_min, row_min, width, height = rect
    return ((col_min + dcol, row_min + drow) \
        for dcol in range(width) \
        for drow in range(height))

def select_rectangles(coords: Coords, rectangles: list[Rectangle]) -> list[Rectangle]:
    """
    Select a coverage of rectangles. It's assumed the list of rectangles are ordered by area,
    from the largest to the smallest
    """
    selected: List[Rectangle] = []
    coverage: Dict[Coord, int] = defaultdict(int) # Number of time a coords is covered. keys=(col, rw), value=(count)

    for rect in rectangles:
        col_min, row_min, width, height = rect
        # If a rectangle doesn't over - overlap ones that has already be selected
        # Add it to the selection and mark its coverage
        if all(coverage[coord] < MAX_OVERLAP or coord not in coords \
            for coord in iter_rect(rect)):
            selected.append(rect)

            for coord in iter_rect(rect):
                coverage[coord] += 1

        # If the shape is entirely covered, the loop can stop
        if all(coverage[coord] > 0 for coord in coords):
            break

    return selected

def orthogonal_set_to_coords(orthoset: OrthogonalSet) -> Coords:
    return set().union(*(set(iter_rect(rect)) for rect in orthoset))

def coords_to_orthogonal_set(coords: Coords) -> tuple[Coords, OrthogonalSet]:
    # Find all the sub rectangles, and order them by area from the largest to the smallest
    rectangles: List[Rectangle] = find_sub_rectangles(coords, False)

    # The use it to extract a coverage:
    orthoset: OrthogonalSet = set(select_rectangles(coords, rectangles))
    coords_remaining: Coords = coords - orthogonal_set_to_coords(orthoset)

    return coords_remaining, set(orthoset)

def find_intersection(rectangles: tuple[Rectangle, ...]) -> Optional[Rectangle]:
    """
    Given a tuple of potentially overlapping rectangles,
    return their intersection if it exists.

    :param rectangles: A tuple of potentially overlapping rectangles
    :return: A rectangle that is their common intersection if it exist
    `None` otherwise
    """
    # To find the intersection,
    # the max_col and the max_row are needed instead of width and height
    boxes: list[Box] = [rectangle_to_box(rect) for rect in rectangles]
    min_cols, min_rows = zip(*[box[0] for box in boxes])
    max_cols, max_rows = zip(*[box[1] for box in boxes])

    min_col, min_row = max(min_cols), max(min_rows) # Min = Max-Min
    max_col, max_row = min(max_cols), min(max_rows) # Max = Min-Max

    # Empty list?
    if not min_col or not min_row or not max_col or not max_row:
        return None

    # No intersection
    if max_col < min_col or max_row < min_row:
        return None

    box_intersection: Box = (min_col, min_row), (max_col, max_row)
    return box_to_rectangle(box_intersection)

def splice_by_removing_a_rectangle(rectangles: list[Rectangle], rectangle) -> list[Rectangle]:
    """
    Given a list of rectangle that potentally overlaps with retcangle,
    Splice them to forms new rectangles, one of which is rectangle, if they contain it.
    Return the newly formed rectangles.
    """
    nrectangles = []
    coords_to_remove = set(iter_rect(rectangle))

    # As the removal is a rectagle on at least an edges, it will create non overlapping
    for rect in rectangles:
        #coords =
        pass

def splice_overlaps(rectangles: list[Rectangle]) -> list[Rectangle]:
    """
    Given a list of recangles pothentially overlapping,
    returns a list of rectangles that don't overlap.
    It finds every intersection, and splice rectangles to make them non-overlapping
    with the overlaps becoming a rectangle of its own.
    Hypothesis: Overlaps are made AT MOST of MAX_OVERLAPS rectangles

    :param rectangles: list of potentially overlapping rectangles
    :return: list of non overlapping rectangles
    """
    without_overlaps = []

    # Check all overlaps by begining by the most important ones
    for i in range(MAX_OVERLAP, 1, -1):
        for combo in combinations(rectangles, i):
            intersection = find_intersection(combo)
            if intersection:
                pass



@debug
def condition_by_rectangles_deprecated(coords: Coords) -> tuple[list[Coords], OrthogonalSet]:
    coords_remaining, orthoset = coords_to_orthogonal_set(coords)
    ncomponents: List[Coords] = condition_by_connected_components(coords_remaining)
    return ncomponents, orthoset

# If there is no area or dimension requirements, they won't be any points remaining
@debug
def condition_by_rectangles(coords: Coords) -> OrthogonalSet:
    coords_remaining, orthoset = coords_to_orthogonal_set(coords)
    ncomponents: List[Coords] = condition_by_connected_components(coords_remaining)
    return orthoset
