"""
The Household and Student classes.
"""
import random
import numpy as np
from .agents_base import BaseAgent
from .agents_spatial import Neighbourhood


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

    _total_households = 0
    _household_utility = np.zeros(61499, dtype="float32")
    __slots__ = ["idx"]

    def __init__(self, unique_id, pos, model, params, category, nhood=None):

        # Store parameters
        super().__init__(unique_id, pos, model, params)

        self.idx = Household._total_households
        Household._total_households += 1

        self.distance = 0
        self.params = params
        self.category = category
        self.school_utility_comp = 0
        self.shape = pos
        self.attributes = self.attribute_array(category)
        self.composition = self.new_composition_array()

        # Create students
        self.students = []
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

    def __repr__(self):
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Household object with unique_id: {self.unique_id}>"

    @property
    def utility(self):
        return Household._household_utility[self.idx]

    @utility.setter
    def utility(self, value):
        Household._household_utility[self.idx] = value

    def attribute_array(self, category):
        """
        This function creates the attribute array for the household that is
        used to calculate the local, neighbourhood and school compositions.

        Args:
            category (int): the category [0,n-1] the agent belongs to. Should
                be generalised in the future.
        """
        attributes = np.zeros(len(self.params['group_types'][0]))
        attributes[category] += 1
        return attributes

    def get_data(self, residential):
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
            self.utility,
            self.category,
            self.unique_id,
            self.distance,

            self.neighbourhood.unique_id if residential
            else self.students[0].school.unique_id,
        ]

        return np.array(data, dtype=float)

    def move_to_empty(self, empties, num_considered, ranking_method):
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

    def residential_move(self, old_pos=None, new_pos=None):
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

    def update(self, residential=True):
        """
        Updates the residential or school composition attributes of the
        Household/Student.

        Args:
            residential (bool): equals True if the model needs to update
                residential or school parameters (default=False).
        """

        if residential:
            self.update_residential()
        else:
            for student in self.students:
                self.update_school(student)

    def update_utilities(self, residential=True):
        """
        Updates the residential or school utility attributes of the
        Household/Student.

        Args:
            residential (bool): equals True if the model needs to update
                residential or school parameters (default=False).
        """

        if residential:
            self.utility = self.model.res_utilities[self.idx]
        else:
            self.school_utility_comp = self.model.school_composition_utilities[
                self.idx]
            self.utility = self.model.school_utilities[self.idx]

    def step(self,
             residential=False,
             initial_schools=False,
             move_allowed=True):
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

    def update_residential(self):
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

    def update_school(self, student):
        """
        Sets the school distance and composition attributes. Note that the
        attributes should only be set here, as this method should only be
        called when the agent actually moves to the location!

        Args:
            student (Student): object to calculate for.

        Todo:
            * Should this be moved to the Student object?
        """

        idx = self.idx

        # Composition utility
        if student.school.total > 0:
            norm = 1.0 / student.school.total
        else:
            norm = 1.0
        self.model.school_compositions[idx] = \
            student.school.composition[self.category] * norm

        # Distance utility
        utility_dist = self.model.distance_utilities[
            self.idx, student.school.idx]

        self.distance = utility_dist
        self.model.distances[idx] = utility_dist

    def residential_utility(self, composition, neighbourhood_composition=None):
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

    def get_closest_neighbourhood(self, pos):
        """
        Find the closest neighbourhood object.

        Returns:
            neighbourhood: the closest (Euclidean) neighbourhood object.
        """
        location = self.model.closest_neighbourhoods[str(pos)]
        neighbourhood = self.model.location_to_agent[location]
        return neighbourhood

    def join_neighbourhood(self, neighbourhood):
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

    def remove_neighbourhood(self, neighbourhood):
        """
        Leave a neighbourhood.

        Args:
            neighbourhood (Neighbourhood): a neighbourhood object.
        """
        neighbourhood.remove_household(self)
        self.neighbourhood = None

    def school_ranking_initial(self):
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

    def residential_ranking(self, positions, ranking_method):
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

    def get_student_count(self):
        """
        Calculates the number of students in a household.

        Returns:
            int: the amount of students in the household.
        """
        return len(self.students)

    def get_shock(self):
        """" Returns a small random float value """
        return self.model.get_shock()

    def get_uniform_shock(self):
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
