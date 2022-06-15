"""
The School class.
"""
from typing import Dict, List, ClassVar
import numpy as np
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
        total (int): the number of Students at the school
        students (dict[int, 'Student']): all the Student objects enrolled in the school.
        composition (array): the sum of the attribute arrays of all Households
            enrolled in this school.
    """

    _total_schools: ClassVar[int] = 0
    __slots__ = ["idx", "total", "capacity", "has_space"]

    @classmethod
    def reset(cls) -> None:
        """
        Resets the school counter.
        """
        cls._total_schools = 0

    def __init__(
            self,
            unique_id: int,
            pos: tuple[float, float],
            model: 'CompassModel',
            params: dict
            ):
        super().__init__(unique_id, pos, model, params)

        self.idx: int = School._total_schools
        School._total_schools += 1

        self.total: int = 0
        self.has_space: bool = True
        self.capacity: int = 1 + int(self.params["school_capacity"] * \
                self.params["n_students"] / self.params["n_schools"])
        self.students: Dict[int, 'Student'] = {}
        self.composition: np.ndarray = self.new_composition_array()

    def __repr__(self) -> str:
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<School object with unique_id: {self.unique_id}>"

    def add_student(self, student: 'Student') -> None:
        """
        Adds a Student object to the list of enrolled students in the School.

        Args:
            student (Student): Student object.
        """
        # Add HOUSEHOLD attributes to the schools' composition
        self.total += 1
        self.composition += student.household.attributes
        self.students[student.idx] = student
        self.has_space = (self.total < self.capacity)

    def remove_student(self, student: 'Student') -> None:
        """
        Removes a Student object from the list of enrolled students in the
        School.

        Args:
            student (Student): Student object.
        """
        # Subtract HOUSEHOLD attributes to the schools' composition
        self.total -= 1
        self.composition -= student.household.attributes  # TODO: zero self.composition?
        self.students.pop(student.idx)
        # after removing a Student, there will always be space
        self.has_space = True

    def get_students(self) -> List['Student']:
        """
        Returns all the students enrolled in the school.

        Returns:
            list: contains all enrolled students.
        """
        return self.students.values()
