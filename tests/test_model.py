import sys
import pytest
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

sys.path.insert(0, "compass")
from parameters import FLAGS
from model import CompassModel
from agents_base import BaseAgent
from agents_spatial import School, Neighbourhood
from agents_household import Household, Student


@pytest.fixture(scope="function")
def model():
    '''
    Returns a CompassModel instance.
    '''
    params = vars(FLAGS)
    model = CompassModel(params)
    return model

# Maybe initialise a BaseAgent instance with random parameter values?
# The scope is now a new instance per function, but ideally hypothesis gets
# a new instance for every time it calls the function?

# def test_grid(model):
#     '''
#     Tests if the grid is initialised correctly.
#     '''

#     households, neighbourhoods, schools, empties = 0, 0, 0, 0
#     agent_ids = set()

#     # Loop over the grid to check its content
#     for agents, x, y in model.grid.coord_iter():

#         if len(agents) == 0:
#             empties += 1

#         else:
#             for object in agents:
#                 # Make sure all ids are unique
#                 agent_ids.add(object.unique_id)

#                 # Count the agent types, there should be no two households in
#                 # the same cell
#                 if isinstance(object, School):
#                     schools += 1

#                 elif isinstance(object, Neighbourhood):
#                     neighbourhoods += 1

#                 elif isinstance(object, Student):
#                     print('No students should be allocated without household')
#                     raise ValueError

#                 else:
#                     households += 1
#                     if len(agents) > 1:
#                         assert not isinstance(object, (Household, Student))

#     assert len(agent_ids) == households + neighbourhoods + schools


def test_distances(model):
    '''
    Loop over every position and check if the distance is not greater than
    the square root of the grid, greater than zero. Also test if nearness is
    bounded between 0 and 1.
    '''
    width = model.params["width"]
    height = model.params["height"]
    max_distance = np.sqrt(width**2 + height**2)
    school_positions = [school.pos for school in model.get_agents('schools')]

    for x in range(width):
        for y in range(height):

            # Distances are supplied in dictionaries of which only the values
            # are checked here.
            distances = model.get_distances((x,y)).values()
            norm_distances =  model.get_norm_distances((x,y))
            # print(distances)
            # print(nearnesses)
            assert len(distances) == len(norm_distances)

            for distance in distances:
                assert distance >= 0
                assert distance <= max_distance

            for agent, norm_dist in norm_distances.items():
                # Only distances to schools are normalised
                if isinstance(agent, School):
                    assert norm_dist >= 0
                    assert norm_dist <= 1


def test_calculate_distance(model):
    distance = model.calculate_distance((2,2), (3,3))
    assert distance >= np.sqrt(2)-0.001
    assert distance <= np.sqrt(2)+0.001


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
