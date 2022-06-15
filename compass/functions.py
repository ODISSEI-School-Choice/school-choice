"""
Some utility functions that cannot be in utils.py due
to circular dependencies.
"""

import numpy as np


def calc_comp_utility(
        dest: np.ndarray,
        x: np.ndarray,
        M: np.ndarray,
        f: np.ndarray
        ) -> None:
    """
    Calculates the utility given a normalised composition (0<=x<=1), an
    optimal fraction (0<=f<=1) and utility at homogeneity (0<=M<=1).

    Arguments
        dest  np.ndarray to hold the output
        x
        M
        f
    """
    dest[x <= f] = (x / f)[x <= f]
    dest[x > f] = (M + (1 - x) * (1 - M) / (1 - f))[x > f]
