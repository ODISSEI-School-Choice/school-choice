"""
This file is used to perform tests while creating the model.
"""
import sys
sys.path.insert(0, "compass")

import random
# import cProfile
import numpy as np
from parameters import FLAGS
from model import CompassModel

# Set seeds for reproducibility
random.seed(3)
np.random.seed(3)


# Initialize
size = 70
FLAGS.n_neighbourhoods = 25
FLAGS.n_schools = 25
# FLAGS.max_res_steps = 10
FLAGS.max_res_steps = 50
# FLAGS.max_school_steps = 100
FLAGS.max_school_steps = 50
FLAGS.width = size
FLAGS.height = size
# FLAGS.conv_threshold = 0.01
FLAGS.conv_threshold = 0
FLAGS.window_size = 30
FLAGS.loglevel = 'DEBUG'
FLAGS.case = 'amsterdam'
# FLAGS.case = 'lattice'
FLAGS.max_move_fraction = 0.05
FLAGS.verbose = True
FLAGS.random_residential = False

if __name__ == '__main__':
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    model = CompassModel(vars(FLAGS), export=True)
    model.simulate()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats(0.05)