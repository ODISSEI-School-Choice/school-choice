from calendar import different_locale
from cmath import exp
from black import diff
import numpy as np
from numpy import random
from sklearn.metrics import ndcg_score


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
    n_households = len(households)
    n_schools = len(schools)

    compositions = []
    for s in schools:
        compositions.append(s.composition)
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

    # TODO: rewrite the following statement
    composition_utilities = np.where(x <= t, x / t, M + (1 - x) * (1 - M) / (1 - t))
    # composition_utilities = np.empty((n_schools, n_households))
    # for i in range(n_schools):
    #     for j in range(n_households):
    #         if x[i, j] <= t[j]:
    #             composition_utilities[i, j] = x[i, j] / t[j]
    #         else:
    #             composition_utilities[i, j] = M[j] + (1 - x[i, j]) * (1 - M[j]) / (
    #                 1 - t[j]
    #             )

    # TODO: rewrite the following statement
    utilities = (
        composition_utilities * alpha[np.newaxis, households_indices]
        + (
            np.take(distance_utilities, households_indices, axis=0)
            * (1 - alpha[households_indices, np.newaxis])
        ).T
    )
    # alpha_selected = np.take(alpha, households_indices)
    # distance_utilities_selected = np.take(
    #     distance_utilities, households_indices, axis=0
    # )
    # utilities = np.empty((n_schools, n_households))
    # for i in range(n_schools):
    #     for j in range(n_households):
    #         utilities[i, j] = composition_utilities[i, j] * alpha_selected[
    #             j
    #         ] + distance_utilities_selected[j, i] * (1 - alpha_selected[j])

    # TODO: rewrite this original version of the code, which is more readable
    schools = np.array(schools)
    for household in households:
        differences = utilities[:, household.idx] - household_utilities[household.idx]
        exp_utilities = np.exp(50 * differences)
        transformed = exp_utilities / exp_utilities.sum()

        ranked_idx = transformed.argsort()[::-1]
        ranking = schools[ranked_idx]
        for student in household.students:
            student.set_school_preference(ranking)
        [student.set_school_preference(ranking) for student in household.students]
    # for j in range(n_households):
    #     exp_sum = 0
    #     for i in range(n_schools):
    #         utilities[i, j] = np.exp(50 * (utilities[i, j] - household_utilities[j]))
    #         exp_sum = exp_sum + utilities[i, j]
    #     for i in range(n_schools):
    #         utilities[i, j] = utilities[i, j] / exp_sum

    #     ranked_indices = utilities[:, j].argsort()[::-1]
    #     ranking = []
    #     for i in range(n_schools):
    #         ranking.append(schools[ranked_indices[i]])
    #     students = households[j].students
    #     n_students = len(students)
    #     for k in range(n_students):
    #         students[k].set_school_preference(ranking)


class Household:
    def __init__(self, idx):
        self.idx = idx
        self.students = [Student()]


class School:
    def __init__(self, idx):
        self.idx = idx
        self.composition = random.rand(2)


class Student:
    def __init__(self):
        self.school_preference = None

    def set_school_preference(self, ranking):
        self.school_preference = ranking
