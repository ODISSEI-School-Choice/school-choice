import sys
import pytest
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from test_functional import random_params

from compass.parameters import FLAGS
from compass.model import CompassModel
from compass.agents_base import BaseAgent
from compass.school import School
from compass.neighbourhood import Neighbourhood
from compass.household import Household
from compass.student import Student


@pytest.fixture(scope="function")
def model():
    '''
    Returns a CompassModel instance.
    '''
    params = vars(FLAGS)
    params['case'] = 'Lattice'
    model = CompassModel(params)
    return model


def test_without_replacement(model):
    """
    Test whether every agent is moved if without_replacement is chosen. Also
    make sure for initial schools everyone is moved.
    """
    scheduler = model.scheduler
    n_splits = scheduler.n_splits
    households = model.get_agents('households')
    unique_agents_stepped = set()

    # Check if all agents are moved for the residential process
    for split in range(int(n_splits)):
        model.step(residential=True, initial_schools=False)
        stepped_agents = scheduler.households_to_move
        print(len(stepped_agents))
        unique_agents_stepped.update(tuple(stepped_agents))
    assert len(households) == len(unique_agents_stepped)

    # Check if all agents are stepped for initial schools
    model.step(residential=False, initial_schools=True)
    households_with_school = [household for household in \
        model.get_agents('households') if household.students[0].school]
    assert len(households) == len(households_with_school)

    # Check if they are all stepped for the school process as well
    unique_agents_stepped = set()
    for split in range(int(n_splits)):
        model.step(residential=False, initial_schools=False)
        stepped_agents = scheduler.households_to_move
        print(len(stepped_agents))
        unique_agents_stepped.update(tuple(stepped_agents))
    assert len(households) == len(unique_agents_stepped)


def test_school_rankings(random_params):
    """
    Test if the rankings contain unique schools equal to the
    total number of schools for 20 random students.    
    """
    print(random_params)
    model = CompassModel(random_params)
    model.simulate(res_steps=10, school_steps=10)
    households = model.get_agents('households')
    np.random.shuffle(households)
    for household in households[:20]:
        for student in household.students:
            ranking = student.school_preference
            # Check if the ranking is equal to the total number
            # of schools
            assert len(ranking) == model.params['n_schools']

            # Check if the school is not in the remainder of the list
            # np.unique() does not work on objects it seems.
            for i, school in enumerate(ranking):
                assert school not in ranking[i+1:]
