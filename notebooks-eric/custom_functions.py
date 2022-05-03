import os
import sys
import SALib
import pickle
import folium
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import contextily as cx
import geopandas as gpd
import matplotlib as mpl
from datetime import date
import multiprocessing as mp
from matplotlib import colors
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from shapely.geometry import Point
from SALib.analyze import sobol, delta
from segregation.singlegroup import Dissim
from itertools import combinations, product
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, "../compass")
from parameters import FLAGS
from model import CompassModel


def save_figure(plot_path, filename):
    # Create directory if it does not exist yet.
    today = date.today()
    _ = plt.savefig(f'{plot_path}/{filename}-{today.strftime("%d-%m-%Y")}.pdf')
    
def extract_data(household_object):
    point = household_object.pos
    category = household_object.category
    return point, category

def allocate(groups, probabilities, actual_households, total_households):
    idx = np.random.choice(total_households, size=actual_households, replace=False)
    allocations = np.array([np.random.multinomial(1, probs) for probs in probabilities])
    return np.matmul(groups[:, idx], allocations[idx,:])

def normalise(distances, alpha):
    decay = 1. / distances**alpha
    normed_decay = decay / decay.sum(axis=1)[:,None]
    return normed_decay

def choice_set(distances, alpha, ind_closest):
    within = (distances < (alpha*1000)).astype(int)
    within[np.arange(within.shape[0]), ind_closest] = 1
    normed_choices = within / within.sum(axis=1)[:,None]
    return normed_choices

def calculate_segregation(data, kind='Theil'):
    compositions = pd.DataFrame(data=data, columns=['group','other_group'])

    if kind.lower()=='theil':

        global_composition = compositions.sum(axis=0)
        global_composition_normalized = global_composition / global_composition.sum()
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

    else:
        compositions['total'] = compositions.sum(axis=1)
        measure = Dissim(compositions, "group", "total").statistic
        return measure