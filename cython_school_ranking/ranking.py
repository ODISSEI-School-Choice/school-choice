import numpy as np
from numpy import random

random.seed(13)


def calc_school_rankings(
    households,
    schools,
    alpha,
    optimal_fraction,
    utility_at_max,
    categories,
    distance_utilities,
    household_utilities,
    dtype,
):
    compositions = []
    for s in schools:
        compositions.append(s.composition_normalized)
    compositions = np.array(compositions, dtype=dtype)

    # get households indices
    households_indices = []
    for h in households:
        households_indices.append(h.idx)
    households_indices.sort()

    # Composition utility calculations
    t = np.take(optimal_fraction, households_indices)
    M = np.take(utility_at_max, households_indices)
    x = np.take(compositions, categories, axis=1)
    composition_utilities = np.where(x <= t, x / t, M + (1 - x) * (1 - M) / (1 - t))

    # Compute utilities
    utilities = (
        composition_utilities * alpha[np.newaxis, households_indices]
        + (
            np.take(distance_utilities, households_indices, axis=0)
            * (1 - alpha[households_indices, np.newaxis])
        ).T
    )

    # Rank schools according to household utilities
    differences = utilities - household_utilities[np.newaxis, :]
    exp_utilities = np.exp(50 * differences)
    transformed = exp_utilities / exp_utilities.sum(axis=0)[np.newaxis, :]
    ranked_indices = transformed.argsort(axis=0)[::-1]

    # Assign ranking to students
    schools = np.array(schools)
    for i in range(len(households)):
        ranking = schools[ranked_indices[:, i]]
        for s in households[i].students:
            s.set_school_preference(ranking)


class Household:
    def __init__(self, idx):
        self.idx = idx
        self.students = [Student()]


class School:
    def __init__(self):
        self.composition_normalized = random.rand(2)


class Student:
    def __init__(self):
        self.school_preference = None

    def set_school_preference(self, ranking):
        self.school_preference = ranking
