import sys
import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from test_functional import random_params

sys.path.insert(0, "compass")
from agents_base import BaseAgent
from agents_spatial import School, Neighbourhood
from agents_household import Household, Student
from model import CompassModel
from parameters import FLAGS


def test_init_household(random_params):
    '''
    Tests if the household is assigned a neighbourhood and the student
    no school.
    '''
    model = CompassModel(random_params)
    household = np.random.choice(model.get_agents('households'))

    # At least one student is in the household
    assert len(household.students) >= 1

    # Check if a neighbourhood is joined initially.
    assert household.neighbourhood != None

    # Check if a school is joined initially (if applicable)
    assert household.students[0].school == None


def test_move_to_empty(random_params):
    '''
    Tests if the household really moves to a spot that was empty before.
    '''
    model = CompassModel(random_params)
    household = np.random.choice(model.get_agents('households'))

    if household.model.params['household_density'] < 1:
        household.move_to_empty(empties=list(household.model.grid.empties),
            num_considered=household.params['num_considered'],
            ranking_method=household.params['ranking_method'])
        cell_content = household.model.grid.get_neighbors(household.pos, 0.001)
        # This cell should have been empty before the move
        assert len(cell_content) == 1
    else:
        assert True


def test_residential_utility(random_params):
    '''
    Tests if the utility of the residential composition is bounded between 0 and 1.
    '''
    model = CompassModel(random_params)
    household = np.random.choice(model.get_agents('households'))
    normalized = household.model.normalized_compositions[
        household.pos[0], household.pos[1], :]
    utility = household.residential_utility(
        composition=normalized,
        neighbourhood_composition=[])
    assert utility <= 1
    assert utility >= 0


def test_school_ranking_initial(random_params):
    '''
    Tests if the INITIAL ranking contains all schools and no duplicates.
    '''
    model = CompassModel(random_params)
    household = np.random.choice(model.get_agents('households'))

    ranking = household.school_ranking_initial()
    # Check if all schools are ranked, and if there are no duplicates
    assert len(ranking) == household.params['n_schools']
    assert len(ranking) == len(set(ranking))


def test_get_student_count(random_params):
    '''
    Tests if every household has at least one student
    '''
    model = CompassModel(random_params)
    for agent in model.get_agents('households'):
        student_count = agent.get_student_count()
        assert student_count >= 1
