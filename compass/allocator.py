"""
The school Allocator class.
"""

class Allocator:
    """
    Class that allocates students across schools given their preferences.
    """

    def __init__(self):
        max_moves = 0

    def optimal_school(self, households):
        """
        Allocate student to their optimal school if they are unsatisfied.

        Args:
            households (list): a list of Household objects.

        Note:
            Utility still based on first student only!!!
        """

        # Loop over all students and schools
        for household in households:
            for student in household.students:
                current_school = student.school
                for school in student.school_preference:

                    # Check if it's the current school
                    if current_school == school:
                        break

                    # Check availability and switch
                    if school.has_space():
                        student.new_school(school)
                        break


    def initial_school(self, households):
        """
        Allocate student to their optimal school.

        Args:
            households (list): a list of Household objects.
        """

        # Loop over all students and schools
        for household in households:
            for student in household.students:
                for school in student.school_preference:

                    # Check availability
                    if school.has_space():
                        student.new_school(school)
                        break
