import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend
import os
import sys
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from propagator import propagator


def _propagator_wrapper(args):
    """Wrapper function for parallel processing"""
    thetap, en, xspan_len = args
    _, temp1, temp2, _ = propagator(thetap, 1j * en)
    return temp1[:xspan_len], temp2[:xspan_len]


def gap_equation(t, delta):
    """
    Calculate the gap profile based on the Green's functions which
    are calculated based on the previous gap profile.

    Parameters:
    -----------
    t : float
        Temperature parameter
    delta : float
        Bulk gap ratio

    Returns:
    --------
    xspan : ndarray
        x coordinate span
    Delta1 : ndarray
        First component of gap profile
    Delta2 : ndarray
        Second component of gap profile
    n : int
        Number of Matsubara frequencies used
    """
    N = 500  # max Matsubara n
    thetapspan = np.linspace(-np.pi / 2, np.pi / 2, 50)

    data = np.loadtxt("gap.txt")
    xspan = data[:, 0]
    xspan_len = len(xspan)

    temp = 0  # store the bulk part of the gap equation
    f1 = np.zeros(xspan_len)
    f2 = np.zeros(xspan_len)
    Delta1 = np.zeros(xspan_len)
    Delta2 = np.zeros(xspan_len)

    for n in tqdm(range(N + 1), desc="Matsubara frequencies"):
        en = (2 * n + 1) * np.pi * t / delta
        temp += (
            1 / np.sqrt(1 + en**2) * np.pi**2 / 2
        )  # bulk part of the gap equation, after the thetap integration

        # Parallel processing over thetap
        args_list = [(thetap, en, xspan_len) for thetap in thetapspan]

        with Pool() as pool:
            results = pool.map(_propagator_wrapper, args_list)

        F1_thetap = np.array([r[0] for r in results])
        F2_thetap = np.array([r[1] for r in results])

        # thetap integral
        F1_integrated = np.trapz(
            np.cos(thetapspan)[:, None] * F1_thetap, thetapspan, axis=0
        )
        F2_integrated = np.trapz(
            np.sin(thetapspan)[:, None] * F2_thetap, thetapspan, axis=0
        )

        f1 += np.real(F1_integrated)
        f2 += np.real(F2_integrated)

        Delta1_new = f1 / temp
        Delta2_new = f2 / temp

        if abs(f2[-1] / temp - Delta2[-1]) < 0.001:
            Delta1 = Delta1_new
            Delta2 = Delta2_new
            break

        Delta1 = Delta1_new
        Delta2 = Delta2_new

    return xspan, Delta1, Delta2, n
