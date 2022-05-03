#! /usr/bin/env python
"""
The BatchRunner class which creates a dataframe with parameter values for
every model run.
"""

import sys
sys.path.insert(0, "compass")
import os
import pickle
import shutil
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from parameters import FLAGS, unparsed

class BatchRunner():
    """
    BatchRunner class which creates a dataframe with parameter values for
    every model run. It also creates a python script which is called repeatedly
    by a bash script called run.sh to allow for the distribution over cores.
    No parallellisation is present within a single model run for now.

    Attributes:
        params (dict): dictionary with all parameter values from FLAGS
        var_params (dict): dictionary with all parameter values to be VARIED
        reps (int): number of repetitions for every configuration
        seed (int): random seed
        n_splits (int): number of splits to distribute over cores
        method (str): one of 'sobol' or 'parameter_sweep' for now
        calc_second_order (bool): calculate second order indices in Sobol method
        n_samples: number of samples to use in the Sobol method
        name (str): prefix for all the filenames
        experiment_path (str): relative path to experiment directory
        data_path (str): relative path to the data folder
        combinations (list): list of parameter combinations
        n_combinations (int): total number of combinations (i.e., model runs)
        problem: specific problem for the Sobol method from SALib
        frame (DataFrame): contains the Dataframe with all model runs
    """

    def __init__(self):

        # Fixed parameters
        params = vars(FLAGS)
        self.params = params

        # Folder structure
        self.name = str(input('Experiment name: '))
        self.experiment_path = f'experiments/{self.name}/'
        self.data_path = f'{self.experiment_path}data/'


        # Check if folders exist, otherwise create them.
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.method = str(input(
            'Method (either parameter_sweep, sobol or convergence_sobol): '))
        self.n_splits = int(input('Number of different jobs: '))
        self.seed = int(input('Random seed: '))
        
        # Method specific settings
        if self.method == 'parameter_sweep':
            self.reps = int(input('Number of repetitions: ')) # Repetitions
            self.interaction_order = int(input('Interaction order: '))

            # PARAMETER_SWEEP VAR_PARAMS
            var_params = {
                        'utility_at_max':[[[x, x]] for x in [
                            0, 0.2, 0.4, 0.6, 0.8, 1]],
                        'optimal_fraction':[[[x, x]] for x in [
                            0.4, 0.45, 0.5, 0.6, 0.7, 0.8]],
                        'alpha':[0, 0.2, 0.4, 0.6, 0.8, 1],
                        'p':[1000, 2000, 4000, 6000],      
                        'q':[1, 2, 4, 6],

                        # 'dist_threshold':[0, 2000, 4000, 6000, 8000, 50000],
                        'random_residential':[0,1],
                        }
                        # ,
                        # 'radius': [1, 3, 5, 7, 9],
                        # 'neighbourhood_mixture':[0, 0.2, 0.4, 0.6, 0.8, 1],
                        # 'max_res_steps':[0, 500]}
            self.convergence_sobol=False

        elif self.method == 'sobol':
            self.calc_second_order = bool(input(
                'Calculate second order (True or False): '))
            self.n_samples = int(input('Number of samples: '))

            # SOBOL VAR_PARAMS
            var_params = {'utility_at_max':[0, 1],
                        'optimal_fraction':[0.4, 0.8],
                        'max_move_fraction':[0, 1],
                        'alpha':[0, 1],
                        'school_capacity':[1, 5],
                        'temperature':[1, 100],
                        'homophily_std':[0, 0.05],
                        'p':[1000, 10000],
                        'q':[1, 10]}
            self.convergence_sobol=False

        elif self.method == 'convergence_sobol':
            # Vary convergence parameters systematically along
            # with Sobol, use only when you know what it does!
            self.convergence_sobol = True
            # SOBOL VAR_PARAMS
            var_params = {'utility_at_max':[0, 1],
                        'optimal_fraction':[0.4, 0.8],
                        'radius':[0, 10],
                        'neighbourhood_mixture':[0, 1],
                        'alpha':[0, 1],
                        'school_capacity':[1, 5],
                        'dummy':[0,1]}
            self.convergence_params = {'num_considered':[1, 5, 10],
                                    'max_move_fraction':[0.25, 0.5, 1],
                                    'scheduling':[0, 1]}
        
        self.var_params = var_params
        

    def create_combinations(self):
        """
        Creates all of the combinations using the given ranges of the parameters
        that are to be varied.
        """
        if self.method=='sobol':
            values = self.saltelli_sequence()
            frame = pd.DataFrame()
            for index, name in enumerate(self.problem['names']):
                
                # The if statement takes care of parameter specific bounds,
                # SALib only allows sampling from a continuous range.
                column = values[:, index]
                if name=='size':
                    frame['width'] = np.floor((50 + (101-50)*column)).astype(int)
                    frame['height'] = frame['width'].copy()

                elif name=='torus':
                    frame[name] = np.round(column).astype(int)

                elif name=='n_schools':
                    squares = np.floor(2 + (8 - 2) * column).astype(int) ** 2
                    frame[name] = squares
                    frame['n_neighbourhoods'] = squares

                elif name=='radius':
                    frame[name] = np.floor(column + 1).astype(int)

                elif name in ['category_thresholds', 'utility_at_max',
                    'optimal_fraction']:
                    frame[name] = [[[x, x]] for x in column]

                elif name=='group_dist':
                    frame[name] = [[[x, 1-x]] for x in column]


                elif name=='num_considered':
                    frame[name] = np.floor(column + 1).astype(int)

                elif name=='window_size':
                    frame[name] = np.floor(column).astype(int)

                elif name=='max_steps':
                    integer = np.floor(column).astype(int)
                    frame['max_res_steps'] = integer
                    frame['max_school_steps'] = integer

                # For the rest of the parameters it's assumed continuous
                else:
                    frame[name] = column

            self.combinations = [dict(row) for index, row in frame.iterrows()]
            self.n_combinations = len(self.combinations)

        else:
            self.combinations = []
            keys = self.var_params.keys()

            # Check if the var params contain only numeric types (see message)
            for key in keys:
                
                if key in ['optimal_fraction', 'utility_at_max']:
                    continue
                
                for value in self.var_params[key]:
                    if type(value) not in [int, float]:
                        print(value)
                        message = ("Only numeric types should be used at first "
                        "because of the check duplicate function. Replace the "
                        "values in num_to_object.")
                        print(message)
                        raise ValueError

            # fill nominal values when not present in combination
            for subset in itertools.combinations(keys, self.interaction_order):

                vals = [self.var_params[var] for var in subset]

                # Iterate over the reps in the outer loop and create combinations of
                # the parameters in the inner one.
                self.combinations += [dict(zip(subset, params)) for params in \
                itertools.product(*vals) for rep in range(self.reps)]
            self.n_combinations = len(self.combinations)


    def saltelli_sequence(self):
        """
        Create the problem and sequence according to SALib specification.
        """
        self.problem = {'num_vars': len(self.var_params.keys()),
                            'names': list(self.var_params.keys()),
                            'bounds': list(self.var_params.values())}

        # Save the problem for later analyses, protocol 4 because the server
        # runs python3.6
        with open(f'{self.experiment_path}problem.pickle', 'wb') as handle:
            pickle.dump(self.problem, handle, protocol=4)

        param_values = saltelli.sample(self.problem, self.n_samples,
            calc_second_order=self.calc_second_order)
        return param_values


    def check_duplicate_runs(self):
        """
        Checks the dataframe for duplicate runs.

        Note:
            * Currently works only for numeric columns
            * Can we extend this function to check in all previous experiments?
        """

        if self.method!='sobol':
            # Drop duplicates only feasible for numeric columns
            filenames = self.frame.filename
            temp_frame = self.frame.select_dtypes(include='number')
            unique_indices = temp_frame.drop_duplicates().index
            self.frame = pd.concat([self.frame.loc[unique_indices]
                ]*self.reps, ignore_index=True)

            # Create new filenames as they are repeated now
            self.frame.filename = filenames[:len(self.frame)]


    def num_to_object(self):
        # Replace scheduling ints with strings
        self.frame.scheduling.replace(
            [0, 1],
            ['replacement', 'without_replacement'],
            inplace=True)
        

    def create_dataframe(self):
        """
        Creates the dataframe, where every row corresponds to one parameter
        configuration for a model run.
        """
        list_of_params = [0]*self.n_combinations

        # If True, these parameters are varied over every Sobol run
        if self.convergence_sobol:
            keys = self.convergence_params.keys()
            order = len(keys)
            conv_combinations = []

            for subset in itertools.combinations(keys, order):
                
                vals = [self.convergence_params[var] for var in subset]
                # Iterate over the reps in the outer loop and create combinations of
                # the parameters in the inner one.
                conv_combinations += [dict(zip(subset, params)) for params in \
                itertools.product(*vals) for rep in range(self.reps)]

            # Repeat the elements until self.n_combinations
            q, r = divmod(self.n_combinations, len(conv_combinations))
            self.conv_combinations =  q * conv_combinations + conv_combinations[:r]


        for index, var_params in enumerate(self.combinations):
            temp_params = self.params.copy()
            # Create a filename
            # filename = f"{self.data_path}{self.name}_{index}"
            filename = f"data/{self.name}_{index}"
            temp_params['filename'] = filename

            if "optimal_fraction" in var_params:
                temp_params["single_optimal_fraction"] = \
                    var_params["optimal_fraction"][0][0]
            else:
                temp_params["single_optimal_fraction"] = \
                    temp_params["optimal_fraction"][0][0]


            if "utility_at_max" in var_params:
                temp_params["single_utility_at_max"] = \
                    var_params["utility_at_max"][0][0]
            else:
                temp_params["single_utility_at_max"] = \
                    temp_params["utility_at_max"][0][0]

            if "category_thresholds" in var_params:
                temp_params["single_category_thresholds"] = \
                    var_params["category_thresholds"][0][0]
            else:
                temp_params["single_category_thresholds"] = \
                    temp_params["category_thresholds"][0][0]

            temp_params.update(var_params)

            if self.convergence_sobol:
                temp_params.update(self.conv_combinations[index])

            list_of_params[index] = temp_params

        self.frame = pd.DataFrame(list_of_params)
        self.check_duplicate_runs()
        self.num_to_object()
        
        print(self.frame.shape)
        print(self.frame.head(25))
        self.frame.hist(figsize=(20,20), align='left', bins=20,
                            backend='matplotlib')

        # Show descriptive statistics
        stats = self.frame.describe()
        for col in stats.columns:
            print()
            print(stats[col])

        plt.savefig(f"{self.experiment_path}histograms-input-data.png")

        # Save under protocol 4, such that it can be opened if using Python3.6
        self.frame.to_pickle(f"{self.experiment_path}{self.name}.pkl",
            protocol=4)


    def split_dataframe(self):
        """
        Creates a list of n_splits Dataframes and saves them.

        Todo:
            What happens when n_splits > len(self.frame)?
        """
        frames = np.array_split(self.frame, self.n_splits)
        self.save_dataframes(frames)


    def save_dataframes(self, frames):
        """
        Saves the dataframes in a folder named 'splits'.

        Args:
            frames (list): list of dataframes.
        """
        path = f"{self.experiment_path}splits/"

        if not os.path.exists(path):
            os.makedirs(path)

        # Delete previous contents
        shutil.rmtree(path[:-1])
        os.makedirs(path)

        for index, frame in enumerate(frames):
            name = f"split{index}"
            frame = frame.reset_index()
            frame.to_pickle(f"{path}{name}.pkl", protocol=4)


    def create_python_script(self):
        """
        Creates the python script used for execution on the server. This script
        is repeatedly called, as fully executing this script is necessary to
        clear the memory every time.
        """
        with open(f'{self.experiment_path}run_batch.py', 'w') as rsh:
            rsh.write('''\
import gc
import sys
sys.path.insert(0, "../../compass")
from model import CompassModel
from parameters import FLAGS, unparsed
import random
import numpy as np
import pandas as pd

# Open df
try:
    jobid = int(unparsed[0])
    row = int(unparsed[1])
    server = True
    # print("Jobid: ", jobid)
except IndexError:
    server = False

if not server:
    frame = pd.read_pickle('{frame}.pkl')
else:
    frame = pd.read_pickle(f"splits/split{{jobid}}.pkl")

model = CompassModel(frame.iloc[row], export=True)
model.simulate()

# for index, row in frame.iterrows():
#     if not server:
#         print("Index: ", index)
#     else:
#         print("Jobid: ", jobid, "Index: ", index)
#     model = CompassModel(row, export=True)
#     model.simulate()
    '''.format(frame=self.name))

        with open(f'{self.experiment_path}check_frame_size.py', 'w') as rsh:
            rsh.write('''\

import sys
import pandas as pd
sys.path.insert(0, "../../compass")
from parameters import FLAGS, unparsed

# Open df
try:
    jobid = int(unparsed[0])
    server = True
except IndexError:
    server = False

if not server:
    frame = pd.read_pickle('{frame}.pkl')
else:
    frame = pd.read_pickle(f"splits/split{{jobid}}.pkl")

print(frame.shape[0] - 1)
    '''.format(frame=self.name))



    def create_bash_script(self):
        """
        Creates the bash script used for execution on the server. It repeatedly
        calls a python script (also created by BatchRunner), as executing a
        python script is necessary to clear the memory every time.
        """
        with open(f'{self.experiment_path}run.sh', 'w') as rsh:
            rsh.write('''\
#!/bin/bash

#SBATCH --job-name={name}
#SBATCH --time=100:00:00

start=0
end=`python3 check_frame_size.py $SLURM_ARRAY_TASK_ID`

for (( i=$start; i<=$end; i++ ))
do
    python3 run_batch.py $SLURM_ARRAY_TASK_ID $i
done
'''.format(name='batchrunner'))




if __name__ == "__main__":
    # Create the batch
    batchrunner = BatchRunner()
    batchrunner.create_combinations()
    batchrunner.create_dataframe()
    batchrunner.split_dataframe()
    batchrunner.create_python_script()
    batchrunner.create_bash_script()
