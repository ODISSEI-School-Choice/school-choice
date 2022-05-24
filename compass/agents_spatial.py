"""
The School and Neighbourhood class.
"""

from .agents_base import BaseAgent


class School(BaseAgent):
    """
    The School class.

    Args:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.

    Attributes:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.
        capacity (float): the maximum amount of students that can be enrolled.
        students (list): all the Student objects enrolled in the school.
        composition (array): the sum of the attribute arrays of all Households
            enrolled in this school.
        normalized_composition (array): same as above but normalized.
    """

    def __init__(self, unique_id, pos, model, params):
        super().__init__(unique_id, pos, model, params)
        self.total = 0
        self.capacity = 1 + int(self.params["school_capacity"] * \
                        self.params["n_students"] / self.params["n_schools"])
        self.students = []
        self.composition = self.new_composition_array()
        self.composition_normalized = self.new_composition_array()

    def __repr__(self):
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<School object with unique_id: {self.unique_id}>"

    def add_student(self, student):
        """
        Adds a Student object to the list of enrolled students in the School.

        Args:
            student (Student): Student object.
        """
        # Add HOUSEHOLD attributes to the schools' composition
        self.total += 1
        self.composition += student.household.attributes
        if self.total == 0:
            self.composition_normalized = self.new_composition_array()
        else:
            self.composition_normalized = self.composition / self.total
        self.students.append(student)

    def remove_student(self, student):
        """
        Removes a Student object from the list of enrolled students in the 
        School.

        Args:
            student (Student): Student object.
        """
        # Subtract HOUSEHOLD attributes to the schools' composition
        self.total -= 1
        self.composition -= student.household.attributes
        if self.total == 0:
            self.composition_normalized = self.new_composition_array()
        else:
            self.composition_normalized = self.composition / self.total
        self.students.remove(student)

    def has_space(self):
        """
        Checks if the school has at least one open spot.

        Returns:
            bool: equals True if there is at least one open spot at the school.
        """
        return len(self.students) < self.capacity

    def get_students(self):
        """
        Returns all the students enrolled in the school.

        Returns:
            list: contains all enrolled students.
        """
        return self.students


class Neighbourhood(BaseAgent):
    """
    The Neighbourhood class.

    Args:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.

    Attributes:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.
        households (list): all the households living in the neighbourhood.
        composition (array): the sum of the attribute arrays of all Households
            belonging to this neighbourhood.
        normalized_composition (array): same as above but normalized.
    """

    def __init__(self, unique_id, pos, shape, model, params):

        super().__init__(unique_id, pos, model, params)
        self.total = 0
        self.shape = shape
        self.households = []
        self.composition = self.new_composition_array()
        self.composition_normalized = self.new_composition_array()

    def __repr__(self):
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Neighbourhood object with unique_id: {self.unique_id}>"

    def add_household(self, household):
        """
        Adds household (not student) object to the list of the Neighbourhood
        object.

        Args:
            household (Household): Household object.
        """
        self.total += 1
        self.composition += household.attributes
        if self.total == 0:
            self.composition_normalized = self.new_composition_array()
        else:
            self.composition_normalized = self.composition / self.total
        self.households.append(household)

    def remove_household(self, household):
        """
        Removes household (not student) object from the list of the
        Neighbourhood object.

        Args:
            household (Household): Household object.
        """
        self.total -= 1
        self.composition -= household.attributes
        if self.total == 0:
            self.composition_normalized = self.new_composition_array()
        else:
            self.composition_normalized = self.composition / self.total
        self.households.remove(household)
