"""
The Household class.
"""
import sys
import random
import numpy as np
from typing import List, ClassVar
from .agents_base import BaseAgent
from .agents_spatial import School
from .student import Student
from .neighbourhood import Neighbourhood

MAX_INITIAL_HOUSEHOLDS = 100


class Household(BaseAgent):
    """
    The household object. Creates an expected number of students per
    household according to the student density, joins the closest
    neighbourhood.

    Args:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.
        category (int): the category [0,n-1] the agent belongs to.

    Attributes:
        unique_id (int): unique identifier of the agent.
        idx (int): index of this Household in lookup arrays
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all (agent) parameter values.
        groups (list): containing all group types.
        attributes (array): array of attributes of the specific agent.
        composition (array): the sum of the attribute arrays of all Households
            in the local composition of this household.
        students (list): the student(s) in the household.
    """

    _total_households: ClassVar[int] = 0
    _max_households: ClassVar[int] = 0

    _household_res_utility: ClassVar[np.ndarray] = np.array([])
    _household_school_utility: ClassVar[np.ndarray] = np.array([])
    _household_distance: ClassVar[np.ndarray] = np.array([])
    _household_school_utility_comp: ClassVar[np.ndarray] = np.array([])
    _household_school_id: ClassVar[np.ndarray] = np.array([])

    __slots__ = ["idx", "category"]

    @classmethod
    def reset(cls, max_households: int = MAX_INITIAL_HOUSEHOLDS) -> None:
        """
        Allocates numpy arrays backing the Household data.

        Must be called before any Household objects are created.

        Arguments:
            max_households (int) : maximum number of Household objects that
            will be created.
        """
        if cls._total_households > 0:
            print("Resetting households while there already are Household"
                  " objects not implemented yet.")
            sys.exit()

        dtype = 'float32'
        cls._max_households = max_households

        # was self.utility if residential
        cls._household_res_utility = np.zeros(max_households, dtype=dtype)

        # was self.utility if not residential
        cls._household_school_utility = np.zeros(max_households, dtype=dtype)

        cls._household_distance = np.zeros(max_households, dtype=dtype)
        cls._household_school_utility_comp = np.zeros(max_households, dtype=dtype)
        cls._household_school_id = np.zeros(max_households, dtype=dtype)

    def __init__(
            self,
            unique_id: int,
            pos: tuple[float, float],
            model: 'CompassModel',
            params: dict,
            category: int,
            nhood: Neighbourhood = None):
        # Store parameters
        super().__init__(unique_id, pos, model, params)

        # Initialize some storage for Household objects
        # Note: this should be done by a call to Household.initialize()
        if Household._max_households == 0:
            print("Household.reset() not called yet, starting with default value")
            Household.reset()

        self.idx: int = Household._total_households
        Household._total_households += 1

        if Household._total_households > Household._max_households:
            print("Too many Household objects!")
            sys.exit(-1)

        self.category: int = category
        self.params: dict = params
        self.shape: tuple[float, float] = pos
        self.attributes: np.ndarray = self.attribute_array()
        self.composition: np.ndarray = self.new_composition_array()
        self.school_id: int = 0

        # Create students
        self.students: List[Student] = []
        for i in range(int(self.params["student_density"])):
            self.students.append(Student(self.model.get_agents("amount"),
                                         self))

        # Join closest neighbourhood if applicable
        if self.params["n_neighbourhoods"]:

            # Join the given neighbourhood or else the closest
            if nhood:
                self.join_neighbourhood(nhood)
            else:
                neighbourhood = self.get_closest_neighbourhood(self.pos)
                self.join_neighbourhood(neighbourhood)

    def __repr__(self) -> str:
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Household object with unique_id: {self.unique_id}>"

    @property
    def school_id(self) -> float:
        return Household._household_school_id[self.idx]

    @school_id.setter
    def school_id(self, value: float) -> None:
        Household._household_school_id[self.idx] = value

    @property
    def res_utility(self) -> float:
        return Household._household_res_utility[self.idx]

    @res_utility.setter
    def res_utility(self, value: float) -> None:
        Household._household_res_utility[self.idx] = value

    @property
    def school_utility(self) -> float:
        return Household._household_school_utility[self.idx]

    @school_utility.setter
    def utility(self, value: float) -> None:
        Household._household_school_utility[self.idx] = value

    @property
    def school_utility_comp(self) -> float:
        return Household._household_school_utility_comp[self.idx]

    @school_utility_comp.setter
    def school_utility_comp(self, value: float) -> None:
        Household._household_school_utility_comp[self.idx] = value

    @property
    def distance(self):
        return Household._household_distance[self.idx]

    @distance.setter
    def distance(self, value):
        Household._household_distance[self.idx] = value

    def attribute_array(self) -> np.ndarray:
        """
        This function creates the attribute array for the household that is
        used to calculate the local, neighbourhood and school compositions.
        """
        attributes = np.zeros(len(self.params['group_types'][0]))
        attributes[self.category] += 1
        return attributes

    def get_data(self, residential: bool) -> np.ndarray:
        """
        Gets the data of a specific agent for storing purposes in utils.py

        Args:
            residential (Bool): True if the model is in the residential process

        Notes:
            Only the school of the first student is used!
        """
        data = [
            self.pos[0], self.pos[1],
            self.composition[0], self.composition[1],

            self.res_utility if residential
            else self.school_utility,

            self.category,
            self.unique_id,
            self.distance,

            self.neighbourhood.unique_id if residential
            else self.students[0].school.unique_id,
        ]

        return np.array(data, dtype=float)

    def move_to_empty(
            self,
            empties: List[tuple[float, float]],
            num_considered: int,
            ranking_method: str
            ) -> None:
        """
        Moves agent to a random empty cell, vacating the agent's old cell.

        Args:
            empties (list): a list of empty coordinates [(x1,y1),..,(xn,yn)]
            num_considered (int): how many spots are considered for the ranking
            ranking_method (str): one of 'highest' or 'proportional'
        """
        if len(empties) == 0:
            raise Exception("ERROR: No empty cells")

        # Pick possible empty locations, rank them and move to the chosen one
        positions = self.random.choices(empties, k=num_considered)
        new_pos = self.residential_ranking(positions, ranking_method)
        self.residential_move(old_pos=self.pos, new_pos=new_pos)

    def residential_move(
            self,
            old_pos: tuple[float, float] = None,
            new_pos: tuple[float, float] = None
            ) -> None:
        """
        Moves a household from old position to its new position.

        Args:
            old_pos (tuple): takes a tuple of integers (x,y), x<width, y<height.
            new_pos (tuple): takes a tuple of integers (x,y), x<width, y<height.
        """

        # Remove the agent from the old neighbourhood and add to the new one
        self.remove_neighbourhood(self.neighbourhood)
        self.model.grid.move_agent(self, new_pos)

        # Only remove the old position from the set if the agent actually
        # moves to a new position
        if new_pos != old_pos:
            self.model.grid.empties.discard(new_pos)
            self.model.grid.empties.add(old_pos)

        neighbourhood = self.get_closest_neighbourhood(self.pos)
        self.join_neighbourhood(neighbourhood)

        # Switch the attributes to the new location as well.
        self.model.switch_attrs(old_pos, new_pos)

    def step(
            self,
            residential: bool = False,
            initial_schools: bool = False,
            move_allowed: bool = True
            ) -> int:
        """
        Steps the agent in the residential or school choice process.

        Args:
            residential (bool): equals True if the model needs to run a
                residential or a school step (default=False).
            initial_schools (bool): equals True if all schools are empty and
                students need an initial allocation first.
            move_allowed (True): equals True if the agent belongs to the
                percentage of agents that is allowed to move.

        Returns:
            int: boolean integer indicating if an agent was moved, to use in
                tracking of moved agents
        """

        # Run advancement for regular residential Schelling model
        if residential:

            # Check if move is allowed
            if not move_allowed:
                return 0

            # Check if there are neighbourhoods to choose from
            if self.params["n_neighbourhoods"] == 0:
                print('There are no neighbourhoods to choose from!')
                return 0

            elif self.params['household_density'] < 1:
                self.move_to_empty(
                    empties=list(self.model.grid.empties),
                    num_considered=self.params['num_considered'],
                    ranking_method=self.params['ranking_method'])

            elif self.params['household_density'] == 1:
                print('Future place for switching of agents.')
                raise NotImplementedError

            return 1

        else:
            # Schools steps are done in the scheduler now for efficiency
            pass

    def update_residential(self) -> None:
        """
        Computes the composition and utility at the current residential location
        and sets the attributes to be used in other calculations.
        """

        category = self.category
        idx = self.idx
        if self.neighbourhood.total > 0:
            norm = 1.0 / self.neighbourhood.total
        else:
            norm = 1.0
        self.model.neighbourhood_compositions[idx] = \
            self.neighbourhood.composition[category] * norm

        if self.params['neighbourhood_mixture'] == 1:
            # Only neighbourhood composition necessary, for case studies and
            # non-integer locations (then integer indexing is not possible)
            self.composition = self.neighbourhood.composition
        else:
            x, y = self.pos
            self.composition = self.model.compositions[x, y, :]
            self.model.local_compositions[idx] = \
                self.model.normalized_compositions[x, y, :][category]

    def residential_utility(
            self,
            composition: np.ndarray,
            neighbourhood_composition: np.ndarray = None
            ) -> float:
        """
        Compute residential utility.

        Args:
            composition (array): normalized local composition counts.
            neighbourhood_composition (array, optional): normalized
                neighbourhood composition counts.

        Returns:
            float: residential utility for a households' current location.

        Note:
            Ideally these computations are executed in numpy arrays,
            simultaneously for all agents, but this is not implemented for
            neighbourhoods yet.
        """
        params = self.params

        if neighbourhood_composition is not None:
            combined = composition*(1-params["neighbourhood_mixture"]) + \
                neighbourhood_composition*params["neighbourhood_mixture"]
        else:
            combined = composition

        actual_fraction = combined[self.category]
        utility_at_max = params["utility_at_max"][0][self.category]
        optimal_fraction = params["optimal_fraction"][0][self.category]

        return self.model.calc_comp_utility(actual_fraction, utility_at_max,
                                            optimal_fraction)

    def get_closest_neighbourhood(self, pos: tuple[float, float]) -> Neighbourhood:
        """
        Find the closest neighbourhood object.

        Returns:
            neighbourhood: the closest (Euclidean) neighbourhood object.
        """
        location = self.model.closest_neighbourhoods[str(pos)]
        neighbourhood = self.model.location_to_agent[location]
        return neighbourhood

    def join_neighbourhood(self, neighbourhood: Neighbourhood) -> None:
        """
        Join the given neighbourhood object.

        Args:
            neighbourhood (Neighbourhood): a neighbourhood object.
        """
        self.neighbourhood = neighbourhood
        neighbourhood.add_household(self)

    def get_neighbourhood(self):
        """
        Return the neighbourhood this household belongs to.
        """
        return self.neighbourhood

    def remove_neighbourhood(self, neighbourhood: Neighbourhood) -> None:
        """
        Leave a neighbourhood.

        Args:
            neighbourhood (Neighbourhood): a neighbourhood object.
        """
        neighbourhood.remove_household(self)
        self.neighbourhood = None

    def school_ranking_initial(self) -> List[School]:
        """
        Computes a list containing all schools ranked to preference. The initial
        school ranking is random.

        Returns:
            list: a randomly ordered list of School objects.
        """

        schools = self.model.get_agents("schools")
        # ranking = np.random.choice(schools, len(schools), replace=False)
        # return ranking

        np.random.shuffle(schools)
        return schools

    def residential_ranking(
            self,
            positions: List[tuple[float, float]],
            ranking_method: str
            ) -> tuple[float, float]:
        """
        Computes the ranked location prefences of a household.

        Args:
            positions (list): list of (x, y) tuples that are considered.
            ranking_method (str): one of 'highest' or 'proportional'

        Returns:
            tuple: new position (x, y) of the household.
        """

        summed = 0
        max_utility = 0
        params = self.params
        positions = list(positions) + [self.pos]  # Append own position
        utilities = np.zeros(len(positions))
        temperature = params['temperature']
        compositions = self.model.compositions
        norm_compositions = self.model.normalized_compositions

        for index, pos in enumerate(positions):

            if pos == self.pos:
                utility = self.utility
            else:
                #  ASSUMING AGENTS HAVE THE SAME RADIUS HERE
                x, y = pos
                composition = compositions[x, y, :]
                norm_composition = norm_compositions[x, y, :]
                neighbourhood = self.get_closest_neighbourhood(pos)
                if neighbourhood.total > 0:
                    norm = 1.0 / neighbourhood.total
                else:
                    norm = 1.0
                utility = self.residential_utility(
                    norm_composition, neighbourhood.composition * norm)

            if utility >= max_utility:
                max_utility = utility
                new_pos = [pos]

            utility = np.exp(temperature * utility)
            summed += utility
            utilities[index] = utility

        utilities = utilities / summed
        if ranking_method == 'proportional' or ranking_method:
            new_pos = random.choices(population=positions,
                                     weights=utilities,
                                     k=1)
        return new_pos[0]

    def get_student_count(self) -> int:
        """
        Calculates the number of students in a household.

        Returns:
            int: the amount of students in the household.
        """
        return len(self.students)

    def get_shock(self) -> float:
        """" Returns a small random float value """
        return self.model.get_shock()

    def get_uniform_shock(self) -> float:
        """
        Returns a random value between 0 and 1.
        """
        return self.model.get_uniform_shock()


class Student():
    """
    Student object that is enrolled into school objects and a Household. Used
    for measuring segregation in schools and neighbourhoods.

    Args:
        unique_id (int): unique identifier of the agent.
        household (Household): Household object.
        groups (list): containing all group types.

    Attributes:
        unique_id (int): unique identifier of the agent.
        household (Household): Household object.
        groups (list): containing all group types.
        school_preference ():
        school (School): School object that the student is enrolled in.
        satisfied (bool): True if the student is satisfied with current school.
        school_history (list): all the school objects the student has attended.

    """

    _total_students = 0

    def __init__(self, unique_id, household):

        self.idx = Student._total_students
        Student._total_students += 1

        self.school = None
        self.school_history = []
        self.unique_id = unique_id
        self.household = household
        self.school_preference = None
        # Student does not inherit from BaseAgent, so increment here.
        self.household.model.increment_agent_count()

    def __repr__(self):
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Student object with unique_id:{self.unique_id}>"

    def set_school_preference(self, ranking):
        """
        Sets school preference to a given ranking.

        Args:
            ranking (list): a ranking of School objects.
        """
        self.school_preference = ranking

    def new_school(self, school):
        """
        Enrolls the student in a new school.

        Args:
            school (School): a School object.
        """
        if self.school:
            self.school.remove_student(self)
        self.school = school
        self.school_history.append(school)
        school.add_student(self)

    def get_school_id(self):
        """
        Returns:
            str: unique id of the school the student is enrolled in.
        """
        if self.school:
            return self.school.unique_id

        return -1
