"""
The Student class.
"""
from typing import List, Iterable


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

    def __init__(self, unique_id: int, household: 'Household'):

        self.idx = Student._total_students  # type: int
        Student._total_students += 1

        self.school = None  # type: 'School'
        self.school_history = []  # type: List['School']
        self.unique_id = unique_id  # type: int
        self.household = household  # type: 'Household'
        self.school_preference = None  # type: Iterable[int]
        # Student does not inherit from BaseAgent, so increment here.
        self.household.model.increment_agent_count()

    def __repr__(self) -> str:
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Student object with unique_id:{self.unique_id}>"

    def set_school_preference(self, ranking: Iterable[int]) -> None:
        """
        Sets school preference to a given ranking.

        Args:
            ranking (list): a ranking of School objects.
        """
        self.school_preference = ranking

    def new_school(self, school: 'School') -> None:
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

    def get_school_id(self) -> int:
        """
        Returns:
            str: unique id of the school the student is enrolled in.
        """
        if self.school:
            return self.school.unique_id

        return -1
