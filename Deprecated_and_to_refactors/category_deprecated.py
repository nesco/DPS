"""
Unused for now.

The Category class is a subcategory of Set.
The aim of this Category module is to give a more structured approach to the standard supervised learning problem.
In supervised learning, there is a set of elements X, a set of elements Y,
and the objective is to learn a mapping f: Xi |-> Yi.

Usually Xis are list of properties called features, Yis are usually simpler, just labels.
Here the problem is considered in its most general form, with both Xis and Yis as set of properties.
No space structure is assumed of their properties. If so, then the properties sharing a common space structure
should be merged in a vector.

The strategy is the following:
    Firstly, try to build a Input Category C(X), which maps how Xis are related to one another.
    It's adds more structure to the Input Space

    Secondly, try to build an Output Category C(Y) out of the output space Y

    Then, for each pair, Pi := {Xi, Yi}, build a category over the pair set C(Pi).
    The pair set tries to map of Xi is transformed into Yi.
    It's only useful if they both have structure and share structural similarities

    Finally, each pair category C(Pi) induce a functor Fi between the input and the output, C(X) and C(Y).
    Those functors F := {Fi}i will thus form a category C(F),
    the morphisms of C(F) being natural transformations of the mapping problem,
    thus the invariants and cliques of C(F) forms a general comprehension of the mapping

    Let's say for now all the inputs to predict are already part of the Input Space X.
    To predict Xk, where Xk doesn't have a pair Yk:
        Extract from C(F) the general pattern of functors:
            On invariants, just copy the corresponding elements of Xk
            On cliques, try to learn the general way an Xi,j is transformed into an Yi,j
"""

from dataclasses import dataclass, field
from collections import defaultdict
from types import MappingProxyType

from typing import TypeVar, Generic, Optional, Union, Callable, Literal, Final, Any
from typing import cast
from itertools import combinations

DEBUG_CATEGORY: bool = False

IndexInvariant = Literal[0]
IndexClique = Literal[1]

INVARIANT: Final[IndexInvariant] = 0
CLIQUE: Final[IndexClique] = 1

ElementClass = Union[IndexInvariant, IndexClique]

T = TypeVar("T")
S = TypeVar("S")

U = TypeVar("U")
V = TypeVar("V")

EndoMap = dict[T, Optional[T]]
LambdaTerm = Callable[[S], Optional[T]]


## Lambda-Calculus Combinators
# Identity Combinator (I)
def IDENTITY(x: T) -> T:
    return x


# Constant Combinator (K)
def CONSTANT(x: T) -> Callable[[Any], T]:
    return lambda y: x


# Substitution Combinator (S)
def SUBSTITUTION(
    x: Callable[[T], Callable[[U], V]],
) -> Callable[[Callable[[T], U]], Callable[[T], V]]:
    return lambda y: lambda z: x(z)(y(z))


# Composition Combinator (B)
def COMPOSITION(x: Callable[[U], V]) -> Callable[[Callable[[T], U]], Callable[[T], V]]:
    return lambda y: lambda z: x(y(z))


# Fixed-point combinator (Y)
def FIXED_POINT(f: Callable[[Callable[[T], U]], Callable[[T], U]]) -> Callable[[T], U]:
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))


##
def PARTIAL_IDENTITY(cut_off: int) -> Callable[[int], Optional[int]]:
    def partial_id(index):
        if index < cut_off:
            return index
        else:
            return None

    return partial_id


IndexCouple = tuple[int, int]  # tuple (i, j)

EnumeratedElement = tuple[int, T]  # tuple (index, element)
IndexElement = tuple[ElementClass, int]

Distance = Union[Callable[[Any, Any], int], Callable[[Any, Any], float]]


@dataclass
class Morphism(Generic[S, T]):
    """
    This class represent a morphism between two objects
    """

    source: set[S]
    target: set[T]
    mapping: LambdaTerm[S, T]

    def __call__(self, element: S) -> Optional[T]:
        return self.mapping(element)

    def compose(self, other: "Morphism") -> "Morphism":
        assert self.target == other.source, (
            "Targets and sources must match for composition."
        )
        return Morphism(
            source=self.source,
            target=other.target,
            mapping=lambda x: other.mapping(self.mapping(x)),
        )

    @staticmethod
    def identity(obj: set[S]):
        return Morphism(source=obj, target=obj, mapping=IDENTITY)


@dataclass
class Category(Generic[T]):
    """
    The Category class represent a full subcategory of the `Set` category.
    Given a list of sets, that will serve as objects, try to build out morphisms using:
        - invariants
        - cliques, that forms as closests objects under a distance or sharing structural properties
        depending of the constructor chosen

    Morphisms here are thus homomorphisms between the sets

    Invariants needs to be treated differently compared to clique. Where invariants directly lead to identity,
    cliques need to lead to an A* search
    """

    # Equivalence classes
    invariants: list[T]  # Elements commpon to all object sets
    cliques: list[
        set[EnumeratedElement]
    ]  # Elements of the different sets that cluster together

    # Traditional categorical definition
    objects: list[set[T]]
    morphisms: dict[IndexCouple, EndoMap] = field(default_factory=dict)

    def element_to_class_index(self, element: T) -> Optional[IndexElement]:
        """
        Given an element, give the index of its equivalence class if it's part of one.
        Equivalence classes are either the element itself if its an invariant,
        or the clique its part of.
        The index of the equivalence class is the index in the list `invariants + cliques`
        """

        if element in self.invariants:
            return (INVARIANT, self.invariants.index(element))

        for i, clique in enumerate(self.cliques):
            if any(element == el for _, el in clique):
                return (CLIQUE, i)

        return None

    def index_to_element(self, index: IndexElement) -> Union[T, set[EnumeratedElement]]:
        # To improve, it's not ok to return cliques'
        element_class, element_index = index
        if element_class == INVARIANT:
            return self.invariants[element_index]
        elif element_class == CLIQUE:
            return self.cliques[element_index]
        else:
            raise ValueError(f"Element Class is invalid: {element_class}")

        # For cliques, you need to get the right el of the cluster, no?


@dataclass
class Category1(Generic[T]):
    """
    The Category class represent a full subcategory of the `Set` category.
    Given a list of sets, that will serve as objects, try to build out morphisms using:
        - invariants
        - cliques, that forms as closests objects under a distance or sharing structural properties
        depending of the constructor chosen

    Morphisms here are thus homomorphisms between the sets

    Invariants needs to be treated differently compared to clique. Where invariants directly lead to identity,
    cliques need to lead to an A* search
    """

    # Equivalence classes
    invariants: list[T]  # Elements commpon to all object sets
    cliques: list[
        set[EnumeratedElement]
    ]  # Elements of the different sets that cluster together

    # Traditional categorical definition
    objects: list[set[T]]
    morphisms: dict[IndexCouple, list[Morphism[set[T], set[T]]]] = field(
        default_factory=dict
    )

    def element_to_class_index(self, element: T) -> Optional[IndexElement]:
        """
        Given an element, give the index of its equivalence class if it's part of one.
        Equivalence classes are either the element itself if its an invariant,
        or the clique its part of.
        The index of the equivalence class is the index in the list `invariants + cliques`
        """

        if element in self.invariants:
            return (INVARIANT, self.invariants.index(element))

        for i, clique in enumerate(self.cliques):
            if any(element == el for _, el in clique):
                return (CLIQUE, i)

        return None

    def index_to_element(self, index: IndexElement) -> Union[T, set[EnumeratedElement]]:
        # To improve, it's not ok to return cliques'
        element_class, element_index = index
        if element_class == INVARIANT:
            return self.invariants[element_index]
        elif element_class == CLIQUE:
            return self.cliques[element_index]
        else:
            raise ValueError(f"Element Class is invalid: {element_class}")

        # For cliques, you need to get the right el of the cluster, no?


class Functor:
    """
    Store a functor between two categories.
    If there is an Input category X and an Output category Y, and between them a pair category i based on one example
    Xi -> Yi.
    Then the morphism m of the pair category will induce functors between Input and Output:

     Funct X -> Y : Xk |-> (Xk -> Xi) • m = (Xi -> Yi) • (Yi -> Yk)
                    (Xk -> Xl) |-> (Yk -> Yl) where
                    Yk = (Xk -> Xi) • (m = (Xi -> Yi)) • (Yi -> Yk)
                    Yl = (Xl -> Xi) • (m = (Xi -> Yi)) • (Yi -> Yl)
    """

    def __init__(self, input_category, output_category, pair_category):
        self.input = input_category
        self.output = output_category
        self.link = pair_category

        self.invariants: list[tuple[IndexElement, IndexElement]] = []
        self.cliques: list[set[EnumeratedElement[IndexElement]]] = []

        self.objects: list[set[IndexElement]] = []
        self.morphisms: dict[IndexCouple, EndoMap[IndexElement]] = field(
            default_factory=dict
        )

        invariants, cliques, morphisms = (
            pair_category.invariants,
            pair_category.cliques,
            pair_category.morphisms,
        )

        # Step 1: Replace input and output sets elements by their class ids in the Input and Output categories
        input_set = pair_category.objects[0]
        output_set = pair_category.objects[1]

        input_id_set = {
            self.input.element_to_class_index(element) for element in input_set
        }
        output_id_set = {
            self.output.element_to_class_index(element) for element in output_set
        }

        self.objects.extend([input_id_set, output_id_set])

        # Step 2: invariants are replaced by an (input_id, output_id) pair

        for element in invariants:
            input_id = input_category.element_to_class_index(element)
            output_id = output_category.element_to_class_index(element)

            self.invariants.append((input_id, output_id))

        # Step 3: elements of cliques are replaced by their id in their respective ategories
        for clique in cliques:
            # Extract elements corresponding to input and output categories
            element_input = next((el for idx, el in clique if idx == 0), None)
            element_output = next((el for idx, el in clique if idx == 1), None)

            # Ensure both elements are found
            if element_input is not None and element_output is not None:
                input_id = input_category.element_to_class_index(element_input)
                output_id = output_category.element_to_class_index(element_output)

                self.cliques.append({(0, input_id), (1, output_id)})

        # Step 4: Same for the morphism
        self.morphisms[(0, 1)] = {}
        for endomap in morphisms.values():
            nendomap = {}

            for input_value, output_value in endomap.items():
                input_class_index = input_category.element_to_class_index(input_value)
                output_class_index = output_category.element_to_class_index(
                    output_value
                )

                if input_class_index is not None:
                    self.morphisms[(0, 1)][input_class_index] = output_class_index


@dataclass
class Functor1(Generic[S, T]):
    source_category: Category[S]
    target_category: Category[T]
    object_mapping: Callable[[int], Optional[int]]  # object mapping
    morphism_mappings: dict[
        IndexCouple, Callable[[int], Optional[int]]
    ]  # morphism mapping


class FunctorCategory(Generic[S, T]):
    """
    Store a functor between two categories.
    If there is an Input category X and an Output category Y, and between them a pair category i based on one example
    Xi -> Yi.
    Then the morphism m of the pair category will induce functors between Input and Output:

     Funct X -> Y : Xk |-> (Xk -> Xi) • m = (Xi -> Yi) • (Yi -> Yk)
                    (Xk -> Xl) |-> (Yk -> Yl) where
                    Yk = (Xk -> Xi) • (m = (Xi -> Yi)) • (Yi -> Yk)
                    Yl = (Xl -> Xi) • (m = (Xi -> Yi)) • (Yi -> Yl)
    """

    def __init__(
        self,
        source_category: Category[S],
        target_category: Category[T],
        pair_category: Category,
    ):
        self.input = source_category
        self.output = target_category
        self.link = pair_category

        self.invariants: list[tuple[IndexElement, IndexElement]] = []
        self.cliques: list[set[EnumeratedElement[IndexElement]]] = []

        self.objects: list[set[IndexElement]] = []
        self.morphisms: dict[IndexCouple, EndoMap[IndexElement]] = field(
            default_factory=dict
        )

        invariants, cliques, morphisms = (
            pair_category.invariants,
            pair_category.cliques,
            pair_category.morphisms,
        )

        # Step 1: Replace input and output sets elements by their class ids in the Input and Output categories
        input_set = pair_category.objects[0]
        output_set = pair_category.objects[1]

        input_id_set = {
            self.input.element_to_class_index(element) for element in input_set
        }
        output_id_set = {
            self.output.element_to_class_index(element) for element in output_set
        }

        self.objects.extend([input_id_set, output_id_set])

        # Step 2: invariants are replaced by an (input_id, output_id) pair

        for element in invariants:
            input_id = input_category.element_to_class_index(element)
            output_id = output_category.element_to_class_index(element)

            self.invariants.append((input_id, output_id))

        # Step 3: elements of cliques are replaced by their id in their respective ategories
        for clique in cliques:
            # Extract elements corresponding to input and output categories
            element_input = next((el for idx, el in clique if idx == 0), None)
            element_output = next((el for idx, el in clique if idx == 1), None)

            # Ensure both elements are found
            if element_input is not None and element_output is not None:
                input_id = input_category.element_to_class_index(element_input)
                output_id = output_category.element_to_class_index(element_output)

                self.cliques.append({(0, input_id), (1, output_id)})

        # Step 4: Same for the morphism
        self.morphisms[(0, 1)] = {}
        for endomap in morphisms.values():
            nendomap = {}

            for input_value, output_value in endomap.items():
                input_class_index = input_category.element_to_class_index(input_value)
                output_class_index = output_category.element_to_class_index(
                    output_value
                )

                if input_class_index is not None:
                    self.morphisms[(0, 1)][input_class_index] = output_class_index


def print_category(category: Category):
    print("Invariants:")
    for invariant in category.invariants:
        print(f"{invariant}")

    for i, clique in enumerate(category.cliques):
        print(f"Clique {i}")
        for l, n in clique:
            print(f"{n}")


def create_lookup_mapping(source_set_index, target_set_index, invariants, cliques):
    """
    Simplest morphism is a straight lookup table.
    """

    def lookup(element):
        if element in invariants:
            return element  # Invariants map to themselves
        else:
            for clique in cliques:
                if (source_set_index, element) in clique:
                    # Find the corresponding element in the target set
                    target_element = next(
                        el for idx, el in clique if idx == target_set_index
                    )
                    return target_element
            return None  # Element doesn't have a mapping

    return lookup


def find_cliques(
    sets: list[set[T]], distance: Distance
) -> list[set[EnumeratedElement]]:
    # Step 1: Try to make associations through pairwise comparisons
    associations = defaultdict(lambda: defaultdict(set))
    for (i, variant_i), (j, variant_j) in combinations(enumerate(sets), 2):
        if not variant_i or not variant_j:
            continue  # No variants to match

        edit_matrix = [[(distance(a, b), a, b) for b in variant_j] for a in variant_i]

        while edit_matrix and any(edit_matrix):
            # min_dist, a, b = max(
            #    (item for row in edit_matrix for item in row if item),
            #        key=lambda x: distance(None, x[1]) + distance(None, x[2]) - x[0]
            #    )
            min_dist, a, b = min(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[0],
            )

            # If adding the association reduces the total cost, add it
            if min_dist < distance(None, a) + distance(None, b):
                associations[i][a].add((j, b))
                associations[j][b].add((i, a))

            # Remove processed items
            edit_matrix = [
                [item for item in row if item[1] != a and item[2] != b]
                for row in edit_matrix
                if any(item[1] != a for item in row)
            ]

    # Step 2: Remove associations that violates transitivity
    # You want to only retains cliques / complete subgraphes
    changed = True
    while changed:
        changed = False
        # For every sets, for each of it's elements
        # Go to the elements it's associated with and check
        # their are only associated with elements its associated with
        for i in associations:
            for a in list(associations[i].keys()):
                for j, b in list(associations[i][a]):
                    for k, c in list(associations[j][b]):
                        # (i, a) is not in itself as the identity transition is not saved
                        if (k, c) not in associations[i][a] and (k, c) != (i, a):
                            # Transitivity violation found, remove the weakest link
                            links = [
                                (
                                    distance(None, a)
                                    + distance(None, b)
                                    - distance(a, b),
                                    (i, a),
                                    (j, b),
                                ),
                                (
                                    distance(None, b)
                                    + distance(None, c)
                                    - distance(b, c),
                                    (j, b),
                                    (k, c),
                                ),
                                (
                                    distance(None, a)
                                    + distance(None, c)
                                    - distance(a, c),
                                    (i, a),
                                    (k, c),
                                ),
                            ]
                            _, (x, y), (z, w) = min(links, key=lambda x: x[0])
                            associations[x][y].discard((z, w))
                            associations[z][w].discard((x, y))
                            changed = True

                            # If an association becomes empty, remove it
                            if not associations[x][y]:
                                del associations[x][y]
                            if not associations[z][w]:
                                del associations[z][w]

    # Step 3: Cluster the clique elements into equivalence classes

    cliques = []
    processed = set()
    for i in associations:
        for a in associations[i]:
            if (i, a) not in processed:
                clique = {(i, a)}
                to_process = [(i, a)]
                while to_process:
                    current = to_process.pop()
                    for associated in associations[current[0]][current[1]]:
                        if associated not in clique:
                            clique.add(associated)
                            to_process.append(associated)

                # Only keep clusters with one element from each set
                if len(clique) == len(sets) and len(set(i for i, _ in clique)) == len(
                    sets
                ):
                    cliques.append(clique)

                processed.update(clique)

    return cliques


def set_to_category2(sets: list[set[T]], distance: Optional[Distance]) -> Category[T]:
    # Hypothesis: distance to None is intrinsic cost
    # If no distance, is given, just find the invariants
    #
    # Step 1: identify the invariants, and extract the variants
    invariants = set.intersection(*sets) if sets else set()
    variants = [s - invariants for s in sets]

    # Step 2: Try to make associations through pairwise comparisons
    associations = defaultdict(lambda: defaultdict(set))
    for (i, variant_i), (j, variant_j) in combinations(enumerate(variants), 2):
        if not variant_i or not variant_j:
            continue  # No variants to match

        edit_matrix = [[(distance(a, b), a, b) for b in variant_j] for a in variant_i]

        while edit_matrix and any(edit_matrix):
            # min_dist, a, b = max(
            #    (item for row in edit_matrix for item in row if item),
            #        key=lambda x: distance(None, x[1]) + distance(None, x[2]) - x[0]
            #    )
            min_dist, a, b = min(
                (item for row in edit_matrix for item in row if item),
                key=lambda x: x[0],
            )

            # If adding the association reduces the total cost, add it
            if min_dist < distance(None, a) + distance(None, b):
                associations[i][a].add((j, b))
                associations[j][b].add((i, a))

            # Remove processed items
            edit_matrix = [
                [item for item in row if item[1] != a and item[2] != b]
                for row in edit_matrix
                if any(item[1] != a for item in row)
            ]

    if DEBUG_CATEGORY and False:
        for i, el in associations.items():
            print(f"Input {i}:")
            for a in el:
                print(f"Element associated to {a}:")
                for j, b in el[a]:
                    print(f"For input {j}: element {b}")
                    print(f"Distance: {distance(a, b)}")
    # Step 3: Remove associations that violates transitivity
    # You want to only retains cliques / complete subgraphes
    changed = True
    while changed:
        changed = False
        # For every sets, for each of it's elements
        # Go to the elements it's associated with and check
        # their are only associated with elements its associated with
        for i in associations:
            for a in list(associations[i].keys()):
                for j, b in list(associations[i][a]):
                    for k, c in list(associations[j][b]):
                        # (i, a) is not in itself as the identity transition is not saved
                        if (k, c) not in associations[i][a] and (k, c) != (i, a):
                            # Transitivity violation found, remove the weakest link
                            links = [
                                (
                                    distance(None, a)
                                    + distance(None, b)
                                    - distance(a, b),
                                    (i, a),
                                    (j, b),
                                ),
                                (
                                    distance(None, b)
                                    + distance(None, c)
                                    - distance(b, c),
                                    (j, b),
                                    (k, c),
                                ),
                                (
                                    distance(None, a)
                                    + distance(None, c)
                                    - distance(a, c),
                                    (i, a),
                                    (k, c),
                                ),
                            ]
                            _, (x, y), (z, w) = min(links, key=lambda x: x[0])
                            associations[x][y].discard((z, w))
                            associations[z][w].discard((x, y))
                            changed = True

                            # If an association becomes empty, remove it
                            if not associations[x][y]:
                                del associations[x][y]
                            if not associations[z][w]:
                                del associations[z][w]

    if DEBUG_CATEGORY and False:
        for i, el in associations.items():
            print(f"Input {i}:")
            for a in el:
                print(f"Element associated to {a}:")
                for j, b in el[a]:
                    print(f"For input {j}: element {b}")
                    print(f"Distance: {distance(a, b)}")

    # Step 4: Cluster the clique elements into equivalence classes

    cliques = []
    processed = set()
    for i in associations:
        for a in associations[i]:
            if (i, a) not in processed:
                clique = {(i, a)}
                to_process = [(i, a)]
                while to_process:
                    current = to_process.pop()
                    for associated in associations[current[0]][current[1]]:
                        if associated not in clique:
                            clique.add(associated)
                            to_process.append(associated)

                # Only keep clusters with one element from each set
                if len(clique) == len(sets) and len(set(i for i, _ in clique)) == len(
                    sets
                ):
                    cliques.append(clique)

                processed.update(clique)

    if DEBUG_CATEGORY and False:
        for i, cluster in enumerate(cliques):
            print(f"\nClique n°{i}")
            for j, el in cluster:
                print(f"- From set n°{j} Element {el}")

    # Step 5: Create morphisms
    morphisms = {}

    for i, j in combinations(range(len(sets)), 2):
        morphisms[(i, j)] = morphisms[(j, i)] = {}

        # Identity on invariants
        for elem in invariants:
            morphisms[(i, j)][elem] = morphisms[(j, i)][elem] = elem

        # Clique-based mapping
        for clique in cliques:
            elem_i = next(a for k, a in clique if k == i)
            elem_j = next(a for k, a in clique if k == j)

            if elem_i is not None and elem_j is not None:
                morphisms[(i, j)][elem_i] = elem_j
                morphisms[(j, i)][elem_j] = elem_i

        # Map remaining variants to None
        for elem in variants[i]:
            if elem not in morphisms[(i, j)]:
                morphisms[(i, j)][elem] = None

        for elem in variants[j]:
            if elem not in morphisms[(j, i)]:
                morphisms[(j, i)][elem] = None

    category = Category(list(invariants), cliques, sets, morphisms)
    return category


def set_to_category(sets: list[set[T]], distance: Optional[Distance]) -> Category[T]:
    # Hypothesis: distance to None is intrinsic cost
    # If no distance, is given, just find the invariants

    # Step 1: identify the invariants, and extract the variants
    invariants = set.intersection(*sets) if sets else set()
    variants = [s - invariants for s in sets]
    cliques: list[set[EnumeratedElement]] = []

    # Step 2: if a distance function is given,
    # try to make clusters out of elements of each sets if they form a clique
    if distance:
        cliques = find_cliques(variants, distance)

    # Collect all elements in invariants and cliques
    selected_elements = invariants.copy()
    for clique in cliques:
        selected_elements.update(elem for _, elem in clique)

    # Find unselected elements for each set
    unselected = [set(s) - selected_elements for s in sets]

    print("Unselected elements:")
    for uns in unselected:
        print(uns)
    # Step 3: Create morphisms
    morphisms = {}

    for i, j in combinations(range(len(sets)), 2):
        morphisms[(i, j)] = morphisms[(j, i)] = {}

        # Identity on invariants
        for elem in invariants:
            morphisms[(i, j)][elem] = morphisms[(j, i)][elem] = elem

        # Clique-based mapping
        for clique in cliques:
            elem_i = next(a for k, a in clique if k == i)
            elem_j = next(a for k, a in clique if k == j)

            if elem_i is not None and elem_j is not None:
                morphisms[(i, j)][elem_i] = elem_j
                morphisms[(j, i)][elem_j] = elem_i

        # Map remaining variants to None
        for elem in variants[i]:
            if elem not in morphisms[(i, j)]:
                morphisms[(i, j)][elem] = None

        for elem in variants[j]:
            if elem not in morphisms[(j, i)]:
                morphisms[(j, i)][elem] = None

    category = Category(list(invariants), cliques, sets, morphisms)
    return category


def set_to_category1(sets: list[set[T]], distance: Optional[Distance]) -> Category[T]:
    # Hypothesis: distance to None is intrinsic cost
    # If no distance, is given, just find the invariants

    # Step 1: identify the invariants, and extract the variants
    invariants = set.intersection(*sets) if sets else set()
    variants = [s - invariants for s in sets]
    cliques: list[set[EnumeratedElement]] = []

    # Step 2: if a distance function is given,
    # try to make clusters out of elements of each sets if they form a clique
    if distance:
        cliques = find_cliques(variants, distance)

    # Step 3: Create morphisms
    morphisms = {}

    for i, j in combinations(range(len(sets)), 2):
        # Create the mapping function from set i to set j
        mapping_i_to_j = create_lookup_mapping(
            source_set_index=i,
            target_set_index=j,
            invariants=invariants,
            cliques=cliques,
        )

        mapping_j_to_i = create_lookup_mapping(
            source_set_index=j,
            target_set_index=i,
            invariants=invariants,
            cliques=cliques,
        )
        # Create the Morphism instances
        lookup_i_to_j = Morphism(source=sets[i], target=sets[j], mapping=mapping_i_to_j)

        lookup_j_to_i = Morphism(source=sets[j], target=sets[i], mapping=mapping_j_to_i)

        morphisms[(i, j)] = morphisms[(j, i)] = {}

        # Store the morphisms in the category's morphisms dictionary
        morphisms[(i, j)] = [lookup_i_to_j]
        morphisms[(j, i)] = [lookup_j_to_i]

    for i, obj in enumerate(sets):
        morphisms[(i, i)] = [Morphism.identity(obj)]

    category = Category(list(invariants), cliques, sets, morphisms)
    return category


def pair_category_to_functor(
    pair_category: Category, source_category: Category, target_category: Category
):
    """
    Transform a pair category {Xi, Yi} into a functor over X and Y.
    Assumptions:
        - source_category.objects and target_category.objects have aligned indices
        Meaning if i is an index in both objects, their objects[i] are in correspondance
        - for each pair (i, j) of indices that exist in both source_category.objects and target_category.objects
        the morphisms of the lists source_category.morphisms[(i, j)] and target_category.morphisms[(i, j)] are
        also aligned by indices.
        Meaning if k is in source_category.morphisms[(i, j)] and target_category.morphisms[(i, j)],
        then source_category.morphisms[(i, j)][k] <> target_category.morphisms[(i, j)][k]
    """

    morphism_mapping = {}

    # Map objects
    if len(source_category.objects) == len(target_category.objects):
        object_mapping = IDENTITY
    else:
        object_mapping = PARTIAL_IDENTITY(len(target_category.objects))

    # Map morphisms
    for (source_idx, target_idx), source_morphisms in source_category.morphisms.items():
        source_idy = object_mapping(source_idx)
        target_idy = object_mapping(target_idx)
        if source_idy is not None and target_idy is not None:
            target_morphisms = target_category.morphisms[(source_idy, target_idy)]
            if len(source_morphisms) == len(target_morphisms):
                morphism_mapping[(source_idx, target_idx)] = IDENTITY
            else:
                morphism_mapping[(source_idx, target_idx)] = PARTIAL_IDENTITY(
                    len(target_morphisms)
                )

            ## DEPRECATED
            # Create a mapping function for the target morphism
            # def create_target_morphism(source_morphism, source_from_obj, source_to_obj, target_from_obj, target_to_obj):
            #    def target_mapping(element_in_target_from_obj):
            # Find the corresponding element in source_from_obj
            corresponding_source_elements = [
                s
                for s in source_from_obj
                if object_mapping.get(s) == element_in_target_from_obj
            ]

            for s in corresponding_source_elements:
                # Apply the source morphism
                s_prime = source_morphism.mapping(s)
                if s_prime is not None:
                    # Map s_prime to the target category
                    t_prime = object_mapping.get(s_prime)
                    if t_prime is not None:
                        return t_prime
            return None  # Element does not have a mapping

    #    target_morphism = Morphism(
    #        source = target_from_obj,
    #        target = target_to_obj,
    #        mapping = target_mapping
    #    )
    #    return target_morphism

    # Iterate over morphisms in the source category
    # for (source_idx_pair), source_morphism in source_category.morphisms.items():
    #    # Unpack source indices
    #    source_from_idx, source_to_idx = source_idx_pair
    #    source_from_obj = source_category.objects[source_from_idx]
    #    source_to_obj = source_category.objects[source_to_idx]

    #    # Get corresponding target objects
    #    target_from_obj = object_mapping[source_from_obj]
    #    target_to_obj = object_mapping[source_to_obj]

    #    target_morphism = create_target_morphism(source_morphism,\
    #        source_from_obj, source_to_obj, \
    #        target_from_obj, target_to_obj)

    #    # Add to morphism_mapping
    #    morphism_mapping[source_morphism] = target_morphism

    functor = Functor(
        source_category, target_category, object_mapping, morphism_mapping
    )
    return functor


def functor_categories_to_natural_transformation_category(
    functor_categories: list[Category],
    category_input: Category[T],
    category_output: Category[T],
) -> Category[T]:
    # Step 1: for morphisms of comma categories, replace their input / output by the ids
    # in category_input / category_output
    morphisms_translated = []
    for functor_category in functor_categories:
        invariants, cliques, morphisms = (
            functor_category.invariants,
            functor_category.cliques,
            functor_category.morphisms,
        )
        nmorphisms = set()
        # Get all the morphisms of each pairs,
        # and write them in a representation that is independant
        # of the specific input and output sets

        print_category(functor_category)
        for endomap in morphisms.values():
            # nendomap = {}
            nendomap = set()

            for input_value, output_value in endomap.items():
                if input_value is None:
                    print("input_value is None")

                if output_value is None:
                    print("output_value is None")
                print(f"input value :{input_value}, output value: {output_value}")

                input_class_index = category_input.element_to_class_index(input_value)
                output_class_index = category_output.element_to_class_index(
                    output_value
                )
                pair_class_index = functor_category.element_to_class_index(input_value)

                if pair_class_index == None:
                    raise ValueError("A morphisms link two non-existant pair objects")

                type, id = pair_class_index

                if input_class_index is not None:
                    # nendomap[input_class_index] = output_class_index
                    # the type of way the morphism is obtained is sufficient
                    nendomap.add((input_class_index, output_class_index, type))

            # nmorphisms.add(frozenset(nendomap.items()))
            nmorphisms.update(nendomap)
        morphisms_translated.append(nmorphisms)

    # Step 2 : extract a category out of the input_output morphisms set
    natural_transformations = set_to_category(morphisms_translated, None)
    return natural_transformations


def functor_categories_to_natural_transformation_category1(
    functor_categories: list[Category],
    category_input: Category[T],
    category_output: Category[T],
) -> Category[T]:
    # Step 1: for morphisms of comma categories, replace their input / output by the ids
    # in category_input / category_output
    morphisms_translated = []
    for functor_category in functor_categories:
        invariants, cliques, morphisms = (
            functor_category.invariants,
            functor_category.cliques,
            functor_category.morphisms,
        )
        nmorphisms = set()
        # Get all the morphisms of each pairs,
        # and write them in a representation that is independant
        # of the specific input and output sets

        for endomap in morphisms.values():
            nendomap = {}

            for element in TODOOOOO:
                # for input_value, output_value in .items():
                input_class_index = category_input.element_to_class_index(input_value)
                output_class_index = category_output.element_to_class_index(
                    output_value
                )
                print(f"Input_class_index: {input_class_index}, value: {input_value}")

                if input_class_index is not None:
                    nendomap[input_class_index] = output_class_index

            nmorphisms.add(frozenset(nendomap.items()))
        morphisms_translated.append(nmorphisms)

    # Step 2 : extract a category out of the input_output morphisms set
    natural_transformations = set_to_category(morphisms_translated, None)
    return natural_transformations
