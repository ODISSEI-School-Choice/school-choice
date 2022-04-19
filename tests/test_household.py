import sys
import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

sys.path.insert(0, "compass")
from agents_base import BaseAgent
from agents_spatial import School, Neighbourhood
from agents_household import Household, Student
from model import CompassModel
from parameters import FLAGS


@pytest.fixture(scope="session")
def household():
    '''
    Returns a Household instance.
    '''
    params = vars(FLAGS)
    model = CompassModel(params)
    random_household = np.random.choice(model.get_agents('households'))
    return random_household


@pytest.fixture(scope="session")
def student(household):
    '''
    Returns a Student instance.
    '''
    student = np.random.choice(household.students)
    return student


# Maybe initialise a BaseAgent instance with random parameter values?
# The scope is now a new instance per function, but ideally hypothesis gets
# a new instance for every time it calls the function?

def test_init_household(household, student):
    '''
    Tests if the household is assigned a neighbourhood and the student
    no school.
    '''
    # At least one student is in the household
    assert len(household.students) >= 1

    # Check if a neighbourhood is joined initially.
    assert household.neighbourhood != None

    # Check if a school is joined initially (if applicable)
    assert student.school == None


def test_move_to_empty(household):
    '''
    Tests if the household really moves to a spot that was empty before.
    '''
    if household.model.params['household_density'] < 1:
        household.move_to_empty(empties=list(household.model.grid.empties),
            num_considered=household.params['num_considered'],
            ranking_method=household.params['ranking_method'])
        cell_content = household.model.grid.get_neighbors(household.pos, 0.001)
        # This cell should have been empty before the move
        assert len(cell_content) == 1
    else:
        assert True


# @given(float1=st.floats(min_value=0, max_value=1),
#         float2=st.floats(min_value=0, max_value=1))
# def test_school_utility(household, float1, float2):
#     '''
#     Tests if the utility is bounded between 0 and 1.
#     '''
#     utility = household.school_utility(comp=float1, dist=float2)
#     assert utility <= 1
#     assert utility >= 0


# @given(int1=st.integers(min_value=0, max_value=10000),
#     int2=st.integers(min_value=0, max_value=10000))
# def test_school_utility_from_composition(household, int1, int2):
#     '''
#     Tests if the utility of the school composition is bounded between 0 and 1.
#     '''
#     normalized = household.model.normalized_compositions[
#         household.pos[0], household.pos[1], :]
#     utility = household.school_utility_from_composition(normalized)
#     assert utility <= 1
#     assert utility >= 0


@given(int1=st.integers(min_value=0, max_value=10000),
    int2=st.integers(min_value=0, max_value=10000))
def test_residential_utility(household, int1, int2):
    '''
    Tests if the utility of the school composition is bounded between 0 and 1.
    '''
    normalized = household.model.normalized_compositions[
        household.pos[0], household.pos[1], :]
    utility = household.residential_utility(
        composition=normalized,
        neighbourhood_composition=[])
    assert utility <= 1
    assert utility >= 0


def test_school_ranking_initial(household, student):
    '''
    Tests if the INITIAL ranking contains all schools and no duplicates.
    '''
    ranking = household.school_ranking_initial(student)
    # Check if all schools are ranked, and if there are no duplicates
    assert len(ranking) == household.params['n_schools']
    assert len(ranking) == len(set(ranking))


# def test_school_ranking(household, student):
#     '''
#     Tests if the ranking contains all schools and no duplicates.
#     '''
#     household.model.step(residential=False, initial_schools=True)
#     household.model.step(residential=False, initial_schools=False)
#     ranking = household.school_ranking(student)
#     assert len(ranking) == household.params['n_schools']
#     assert len(ranking) == len(set(ranking))


def test_get_student_count(household):
    '''
    Tests if every household has at least one student
    '''
    for agent in household.model.get_agents('households'):
        student_count = agent.get_student_count()
        assert student_count >= 1
