#! /usr/bin/env python
"""
The DataCrawler class which loops over BatchRunner runs to perform calculations.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from SALib.sample import saltelli
from itertools import combinations, product
from segregation.singlegroup import CorrelationR, Dissim, Entropy, Isolation

class DataCrawler:

    def __init__(self, dataframe, path_to_frame):
        """
        This class is used to loop over all datasets and gather data

            dataframe (DataFrame): consists of all parameter values and
                filenames. Every row should represent a model run.

        """
        self.dataframe = dataframe
        self.path = path_to_frame
        self.gather_data()


    def open_file(self, filename):
        """
        Tries to open filename.npz and stores the data in the corresponding
        attributes.

        Args:
            filename (str): filename
        """
        data = np.load(filename)
        # The file contains arrays for households, schools and neighbourhoods
        for agents in ['households', 'neighbourhoods', 'schools']:

            temp_data = data[agents]
            headers = data[agents + '_headers']
            time, number, attributes = temp_data.shape
            iterables = [range(time), range(number)]

            # Create DataFrame with MultiIndex for easy indexing
            index = pd.MultiIndex.from_product(iterables,
                                                names=["time", "agent"])
            frame = pd.DataFrame(data=temp_data.reshape(time*number, attributes),
                                columns=headers, index=index)

            # Note that residential and school time both have different t=0
            if agents == 'households':
                self.households = frame
            elif agents == 'neighbourhoods':
                self.neighbourhoods = frame
                self.last_res_step = time - 1
            elif agents == 'schools':
                self.schools = frame
                self.last_school_step = time - 1

    
    def calculate_segregation(self, data, kind="isolation"):
        """
        Calculates the isolation index from the segregation package.

        Args:
            data (DataFrame): DataFrame where every row corresponds to an
                organisational unit (e.g., school or neighbourhood). Columns
                should contain the absolute counts of groups.

        Todo:
            *Calculation gives weird results
            *Column names are hardcoded for now, this should change.
        """
        temp_data = data.copy()
        temp_data['totals'] = temp_data['comp_0'] + temp_data['comp_1']
        if kind.lower()=='entropy':
            statistic = Entropy(temp_data,'comp_0','totals').statistic
        elif kind.lower()=='correlation':
            statistic = CorrelationR(temp_data,'comp_0','totals').statistic
        elif kind.lower()=='dissim':
            statistic = Dissim(temp_data,'comp_0','totals').statistic
        else:
            statistic = Isolation(temp_data,'comp_0','totals').statistic

        return statistic


    def calculate_theil(self, data):
        """
        Calculate Theil's information index (own implementation).

        Args:
            data (DataFrame): DataFrame where every row corresponds to an
                organisational unit (e.g., school or neighbourhood). Columns
                should contain the absolute counts of groups.

        Todo:
            *Column names are hardcoded for now, this should change.
        """

        compositions = data[['comp_0', 'comp_1']]
        global_composition = compositions.sum(axis=0)
        global_composition_normalized = global_composition / \
                global_composition.sum()
        pi_m = global_composition_normalized.values

        t_j = compositions.sum(axis=1)
        pi_jm = compositions.divide(t_j, axis=0)
        T = t_j.sum()
        r_jm = pi_jm / pi_m

        global_entropy = - np.sum(pi_m * np.log(pi_m))
        E = global_entropy
        log_r_jm = np.nan_to_num(np.log(r_jm))

        H = np.sum((t_j / (T * E)) * (pi_jm * log_r_jm).T)
        theil = H.sum()
        return theil


    def save_dataframe(self, filename, dataframe):
        """
        Saves dataframe under filename.pickle.
        """
        dataframe.to_pickle(filename)


    def process_file(self, iterrows):
        """
        This is where additional calculations can be made on the dataset of all
        model runs from the batchrunner (per model run).

        Args:
            iterrows: a DataFrame iterrows object (containing index and row).
        """
        index, row = iterrows
        params = row.copy()
        filename = params['filename'] + '.npz'
        try:
            self.open_file(filename)
        except FileNotFoundError:
            print('File not found: ', filename)
            return None

        if not params['save_last_only']:
            # Calculate Theil in first step for randomness correction purposes
            params['res_seg_init'] = self.calculate_theil(self.neighbourhoods.xs(0))
            params['school_seg_init'] = self.calculate_theil(self.schools.xs(0))

            # Check convergence
            params['res_converged'] = self.last_res_step+1<params['max_res_steps']
            params['school_converged'] = self.last_school_step+1<params['max_school_steps']
        else:
            # Only last step is saved, so the index to be used is 0
            # households are saved a bit different, but not used in these
            # calculations for now
            self.last_res_step = self.last_school_step = 0

        # Calculate segregation using our own implementation of Theil's
        params['residential_own_entropy'] = self.calculate_theil(
            self.neighbourhoods.xs(self.last_res_step))
        params['school_own_entropy'] = self.calculate_theil(
            self.schools.xs(self.last_school_step))

        for kind in ['entropy', 'correlation', 'dissim', 'isolation']:
            params['residential_' + kind] = self.calculate_segregation(
                self.neighbourhoods.xs(self.last_res_step), kind=kind)
            params['school_' + kind] = self.calculate_segregation(
                self.schools.xs(self.last_school_step), kind=kind)

        return params


    def gather_data(self):
        """
        Distributes gathering all the data from all model runs over the maximum
        number of cores available.
        """
        start = time.time()
        chunksize = 100
        with mp.Pool() as p:
            list_of_params = list(tqdm(p.imap(
                self.process_file, self.dataframe.iterrows(),
                chunksize=chunksize), total=self.dataframe.shape[0]))
        end = time.time()
        print("Time elapsed:", end - start)

        # Save the dataframe in a new file
        frame = pd.DataFrame(list_of_params)
        self.save_dataframe(f"data/frame.pkl", frame)


if __name__ == "__main__":

    name = str(input('Experiment to crawl: '))
    path_to_frame = f'experiments/{name}/'
    print(os.getcwd())
    print(f"{name}.pkl")
    frame = pd.read_pickle(f"{name}.pkl")
    crawler = DataCrawler(frame, path_to_frame)
    segregation_data = pd.read_pickle(f"data/frame.pkl")
