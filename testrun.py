"""
This file is used to perform tests while creating the model.
"""
import sys
sys.path.insert(0, "compass")


import random
import cProfile
import numpy as np
from parameters import FLAGS
from model import CompassModel

# Set seeds for reproducibility
random.seed(3)
np.random.seed(3)


# Initialize
size = 50
FLAGS.n_neighbourhoods = 25
FLAGS.n_schools = 25
FLAGS.max_res_steps = 0
FLAGS.max_school_steps = 100
FLAGS.width = size
FLAGS.height = size
FLAGS.conv_threshold = 0.0
FLAGS.window_size = 30
FLAGS.loglevel = 'DEBUG'
FLAGS.case = 'IJburg'
FLAGS.max_move_fraction = 0.25

if __name__ == '__main__':
    # import cProfile, pstatsP
    # profiler = cProfile.Profile()
    # profiler.enable()
    model = CompassModel(vars(FLAGS), export=True)
    model.simulate()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats(0.05)