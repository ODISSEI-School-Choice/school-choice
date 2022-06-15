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
            # we need to update the household with information of the first
            # student only. Keep track if we've done that using this variable:
            do_household_update = True
            start_at_school = 0

            for student in household.students:
                current_school = student.school
                for school in student.school_preference[start_at_school:]:

                    # Check if it's the current school
                    if current_school == school:
                        # break means the student stays at its current school
                        break

                    # Check availability and switch
                    if school.has_space:
                        student.new_school(school)
                        # update household
                        if do_household_update:
                            model = household.model

                            # link household and school for the first student
                            # in household_data we need household.students[0].school.unique_id
                            household.school = school
                            household.school_id = school.unique_id
                            household.distance = model.distance_utilities[
                                    household.idx, school.idx
                                    ]
                            do_household_update = False
                        break
                    # As student from the same household have the same preference,
                    # we do not need to check for space in this school anymore
                    start_at_school += 1


    def initial_school(self, households):
        """
        Allocate student to their optimal school.

        Args:
            households (list): a list of Household objects.
        """

        # Loop over all students and schools
        for household in households:
            # we need to update the household with information of the first
            # student only. Keep track if we've done that using this variable:
            do_household_update = True
            start_at_school = 0
            for student in household.students:
                for school in student.school_preference[start_at_school:]:

                    # Check availability
                    if school.has_space:
                        student.new_school(school)
                        # update household
                        if do_household_update:
                            model = household.model

                            # link household and school for the first student
                            # in household_data we need household.students[0].school.unique_id
                            household.school = school
                            household.school_id = school.unique_id
                            household.distance = model.distance_utilities[
                                    household.idx, school.idx
                                    ]
                            do_household_update = False
                        break
                    # As student from the same household have the same preference,
                    # we do not need to check for space in this school anymore
                    start_at_school += 1
