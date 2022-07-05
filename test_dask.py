from copy import deepcopy
from dask.distributed import Client
from compass.model import CompassModel
from compass.parameters import FLAGS

# Initialize
FLAGS.size = 50
FLAGS.max_res_steps = 10
FLAGS.max_school_steps = 10
FLAGS.conv_threshold = 0.01
FLAGS.window_size = 30
FLAGS.max_move_fraction = 0.05
FLAGS.verbose = False
FLAGS.random_residential = False
FLAGS.visualisation = False
FLAGS.case = "lattice"
FLAGS.filename = 'vis_data'
FLAGS.alpha = 1
FLAGS.seed = 3  # Runs use a single RNG, and are exactly reproducible


def delayed_run(opt_frac):
    """Perform a single model run."""
    run_flags = deepcopy(FLAGS)
    run_flags.optimal_fraction = [[opt_frac, opt_frac]]
    model = CompassModel(**vars(run_flags), export=False)
    model.simulate()
    return model.segregation[-1]


if __name__ == "__main__":
    # Setup dask and
    # processes = True       : make sure each Model has its own data structures
    # threads_per_worker = 1 : really, dont try multithreading
    client = Client(processes=True, threads_per_worker=1)

    delayed_results = client.map(delayed_run, [0.4, 0.5, 0.6, 0.7, 0.8])
    results = client.gather(delayed_results)
    client.close()
    print(results)
