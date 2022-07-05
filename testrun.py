"""
This file is used to perform tests while creating the model.
"""
import random
import numpy as np
from compass.model import CompassModel
from compass.parameters import FLAGS

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
FLAGS.loglevel = "DEBUG"
FLAGS.case = "amsterdam"
# FLAGS.case = 'lattice'
FLAGS.max_move_fraction = 0.05
FLAGS.verbose = True
FLAGS.random_residential = False
FLAGS.seed = 3

if __name__ == "__main__":
    model = CompassModel(**vars(FLAGS), export=True)
    model.simulate()
    print(model.segregation[-1])
