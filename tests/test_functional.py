import sys
import pytest
import logging
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

sys.path.insert(0, "compass")
from parameters import FLAGS
from model import CompassModel
from agents_base import BaseAgent
from agents_household import Household, Student
from agents_spatial import School, Neighbourhood

log = logging.getLogger()

def log_params(params):
    logging.warning(f"\nParameters: {params}")


def log_steps(model):
    res_time = model.scheduler.get_time('residential')
    school_time = model.scheduler.get_time('school')
    logging.warning(f"\nResidential steps: {res_time}/{model.params['max_res_steps']}")
    logging.warning(f"\nSchool steps: {school_time}/{model.params['max_school_steps']}")


def log_segregation(model):
    res_seg = model.measurements.calculate_segregation(
        type="bounded_neighbourhood", index="Theil")
    school_seg = model.measurements.calculate_segregation(
        type="school", index="Theil")
    logging.warning(f"\nResidential segregation: {res_seg}")
    logging.warning(f"\nSchool segregation: {school_seg}")
    return res_seg, school_seg

@pytest.fixture(scope="function")
def random_params():
    '''
    Returns random parameter values, where bounds for the parameters are given,
    such that the tests do not run for too long.
    '''
    params = vars(FLAGS)

    # Draw random parameters
    params['loglevel'] = 'DEBUG'
    params['width'] = np.random.randint(40, 70)
    params['height'] = params['width']
    params['alpha'] = np.random.uniform()
    params['torus'] = np.random.choice([0,1])
    params['household_density'] = np.random.uniform(0.50, 0.90)
    params['max_move_fraction'] = np.random.uniform(0.1, 0.5)
    params['max_res_steps']= 200 # should be sufficiently large
    params['max_school_steps']= 200 # should be sufficiently large
    params['conv_threshold'] = np.random.uniform(0.005, 0.01)
    params['n_neighbourhoods'] = np.random.choice(
        [x ** 2 for x in range(2, 8)])
    params['n_schools'] = np.random.choice(
        [x ** 2 for x in range(2, 8)])

    rand1, rand2 = np.random.uniform(low=0, high=1, size=2)
    params['group_dist'] = [[0.5, 0.5]]
    params['utility_at_max'] = [[rand2, rand2]]
    rand3 = np.random.uniform(low=0.5, high=0.9)
    params['optimal_fraction'] = [[rand3, rand3]]

    params['school_capacity'] = 2
    params['radius'] = np.random.randint(1, 10)
    params['neighbourhood_mixture'] = np.random.uniform()
    params['temperature'] = np.random.uniform(10, 100)
    params['homophily_std'] = np.random.uniform(0.001, 0.1)
    params['window_size'] = 20
    params['num_considered'] = np.random.choice(range(1, 2))
    params['ranking_method'] = 'highest'

    params['case'] = 'lattice' # Amsterdam case is not tested yet

    return params


def test_distance_only(random_params):
    '''
    Only distance matters in school choice, so residential segregation should
    equal school segregation and the distribution of school utilities should
    equal that of the nearnesses.
    '''
    logging.warning('\n\n\nDISTANCE ONLY')
    random_params['alpha'] = 0
    random_params['n_schools'] = random_params['n_neighbourhoods']
    log_params(random_params)
    model = CompassModel(random_params)
    model.simulate(res_steps=random_params['max_res_steps'],
                    school_steps=random_params['max_school_steps'])
    log_steps(model)
    res_seg, school_seg = log_segregation(model)
    if res_seg>0.1:
        assert np.isclose(res_seg, school_seg, rtol=0.05)


def test_composition_only(random_params):
    '''
    Only composition matters in school choice, so in the mildest case where
    everyone chooses the closest school or preferences are very tolerant,
    we would see school segregation is roughly equal to residential segregation.
    Else it should be higher.
    '''
    logging.warning('\n\n\nCOMPOSITION ONLY')
    random_params['alpha'] = 1
    random_params['n_schools'] = random_params['n_neighbourhoods']
    log_params(random_params)
    model = CompassModel(random_params)
    model.simulate(res_steps=random_params['max_res_steps'],
                    school_steps=random_params['max_school_steps'])
    log_steps(model)
    res_seg, school_seg = log_segregation(model)
    if (res_seg>0.1 and res_seg<0.95):
        if school_seg >= res_seg:
            assert school_seg >= res_seg
        else:
            assert np.isclose(res_seg, school_seg, rtol=0.05)
    else:
        assert 1


def test_bounded_racist(random_params):
    '''
    People only look at bounded neighbourhoods and are only satisfied with
    homogeneity (neighbhourhoods / schools of one type). This should lead to
    both types of segregation being close to 1 (fully segregated).
    '''
    logging.warning('\n\n\nBOUNDED RACIST')
    random_params['neighbourhood_mixture'] = 1
    random_params['homophily_std'] = 0
    random_params['optimal_fraction'] = [[1, 1]]
    random_params['n_schools'] = random_params['n_neighbourhoods']
    log_params(random_params)
    model = CompassModel(random_params)
    model.simulate(res_steps=random_params['max_res_steps'],
                    school_steps=random_params['max_school_steps'])
    log_steps(model)
    res_seg, school_seg = log_segregation(model)
    assert np.isclose(res_seg, 1, atol=0.1)
    assert np.isclose(school_seg, 1, atol=0.1)
