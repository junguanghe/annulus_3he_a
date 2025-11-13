import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from propagator import propagator


def _propagator_wrapper(
    args: tuple[float, float, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper function for parallel processing"""
    thetap, en, xspan, Delta1, Delta2 = args
    _, temp1, temp2, _ = propagator(thetap, 1j * en, xspan, Delta1, Delta2)
    return temp1[: len(xspan)], temp2[: len(xspan)]


def gap_equation(
    t: float,
    delta: float,
    xspan: np.ndarray,
    Delta1_old: np.ndarray,
    Delta2_old: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate the gap profile based on the Green's functions which
    are calculated based on the previous gap profile.

    Parameters:
    -----------
    t : float
        Temperature parameter
    delta : float
        Bulk gap ratio
    xspan : ndarray
        x coordinate span from previous iteration
    Delta1_old : ndarray
        First component of gap profile from previous iteration
    Delta2_old : ndarray
        Second component of gap profile from previous iteration

    Returns:
    --------
    xspan : ndarray
        x coordinate span (same as input)
    Delta1 : ndarray
        First component of gap profile
    Delta2 : ndarray
        Second component of gap profile
    n : int
        Number of Matsubara frequencies used
    """
    N = 500  # max Matsubara n
    thetapspan = np.linspace(-np.pi / 2, np.pi / 2, 50)

    xspan_len = len(xspan)

    temp = 0  # store the bulk part of the gap equation
    f1 = np.zeros(xspan_len)
    f2 = np.zeros(xspan_len)

    for n in tqdm(range(N + 1), desc="Matsubara frequencies"):
        en = (2 * n + 1) * np.pi * t / delta
        temp += (
            1 / np.sqrt(1 + en**2) * np.pi**2 / 2
        )  # bulk part of the gap equation, after the thetap integration

        # Parallel processing over thetap
        args_list = [
            (thetap, en, xspan, Delta1_old, Delta2_old) for thetap in thetapspan
        ]

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

        if abs(Delta2_new[-1] - Delta2_old[-1]) < 0.001:
            break

        Delta1_old = Delta1_new
        Delta2_old = Delta2_new

    return xspan, Delta1_new, Delta2_new, n
