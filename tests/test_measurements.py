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

@pytest.fixture(scope="function")
def model():
    '''
    Returns a CompassModel instance.
    '''
    params = vars(FLAGS)
    model = CompassModel(params)
    return model

'''
Needs to be implemented still, but maybe it's better to wait after the
segregation package is used. Then we need to pass the data a bit different.
'''

# Maybe initialise a BaseAgent instance with random parameter values?
# The scope is now a new instance per function, but ideally hypothesis gets
# a new instance for every time it calls the function?


# def test_segregation(model):
#     '''
#     Tests if the segregation measures are calculated correctly in
#     certain special cases.
#     '''
#     neighbourhoods = model.get_agents('neighbourhoods')
#     for neighbourhood in neighbourhoods:
#         # First delete all the households
#         for household in neighbourhood.households:
#             model.scheduler.remove(household)
#
#         model.grid.place_agent(household, neighbourhood.pos)
#         model.scheduler.add(household)
#         neighbourhood = household.find_closest_neighbourhood()
#         household.join_neighbourhood(neighbourhood)
#
#     model.global_composition_normalized = model.normalize_composition(
#         model.global_composition)
#     types = ['school', 'bounded_neighbourhood', 'local_neighbourhood']
#     measured = [0]*len(types)
#     for index, type in enumerate(types):
#         measured[index] = model.measurements.calculate_theil(type)
#     print(measured)
#     assert np.all(np.array(measured) > 0.999)
