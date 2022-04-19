import sys
import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

sys.path.insert(0, "compass")
from agents_base import BaseAgent
from model import CompassModel
from parameters import FLAGS
import contextlib

@contextlib.contextmanager
@pytest.fixture(scope="session")
def base_agent():
    '''
    Returns a BaseAgent instance.
    '''
    params = vars(FLAGS)
    model = CompassModel(params)
    object = BaseAgent(None, None, model, params)
    return object

# Maybe initialise a BaseAgent instance with random parameter values?
# The scope is now a new instance per function, but ideally hypothesis gets
# a new instance for every time it calls the function?

def test_new_composition_array(base_agent):
    '''
    Tests if the new composition dictionary is configured correctly.
    '''
    for share in base_agent.new_composition_array():
        assert share == 0.


# def test_normalize_composition_array(base_agent):
#     '''
#     Tests if the normalized composition returns equal shares.
#     '''
#     composition = np.array([1, 1])
#     for share in base_agent.normalize_composition_array(composition):
#         assert share == 0.5


# @given(int1=st.integers(min_value=0, max_value=10000),
#     int2=st.integers(min_value=0, max_value=10000))
# def test_normalize_composition_array_extended(base_agent, int1, int2):
#     '''
#     Tests if the normalised composition is 0<=x<=1 for several integers>=0.
#     '''
#     composition = np.array([int1, int2])
#     normalized_composition = base_agent.normalize_composition_array(composition)
#     print(composition, normalized_composition)
#     for share in normalized_composition:
#         assert share <= 1
#         assert share >= 0
