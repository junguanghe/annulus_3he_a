from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d


def propagator(
    thetap: float,
    epsilon: complex,
    xspan: np.ndarray,
    Delta1: np.ndarray,
    Delta2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Green's function of a chiral p-wave superconductor near a wall.

    Parameters:
    -----------
    thetap : float
        Angle parameter
    epsilon : complex
        Energy in unit of bulk gap Delta_0, a typical value is 1j*(2*n+1)*pi*0.1/1.7639
    xspan : ndarray
        x coordinate span
    Delta1 : ndarray
        First component of gap profile
    Delta2 : ndarray
        Second component of gap profile

    Returns:
    --------
    x : ndarray
        x coordinate of the classical trajectory
    f1 : ndarray
        First component of the Green's function
    f2 : ndarray
        Second component of the Green's function
    g : ndarray
        Diagonal component of the Green's function
    """

    px = np.cos(thetap)
    py = np.sin(thetap)
    x = np.concatenate([xspan, -np.flip(xspan[:-1])])

    if abs(thetap - np.pi / 2) < 1e-10 or abs(thetap + np.pi / 2) < 1e-10:
        f1 = np.zeros(len(x))
        f2 = np.pi * py / np.sqrt(1 - epsilon**2) * np.ones(len(x))
        g = -epsilon * np.pi / np.sqrt(1 - epsilon**2) * np.ones(len(x))
        return x, f1, f2, g

    # encode a '-' from -px after bouncing off the wall for Delta1
    Delta1 = np.concatenate([px * Delta1, -px * np.flip(Delta1[:-1])])
    Delta2 = np.concatenate([py * Delta2, py * np.flip(Delta2[:-1])])

    # Interpolation
    interp_Delta1 = interp1d(x, Delta1, kind="cubic", fill_value="extrapolate")
    interp_Delta2 = interp1d(x, Delta2, kind="cubic", fill_value="extrapolate")

    xq = (x[:-1] + x[1:]) / 2  # half step (used in 4th-order Runge Kutta method)
    delta1 = interp_Delta1(xq)
    delta2 = interp_Delta2(xq)

    a = np.zeros(len(x), dtype=complex)
    abar = np.zeros(len(x), dtype=complex)

    g = -np.pi * epsilon / np.sqrt(1 - epsilon**2)
    f1 = np.pi / np.sqrt(1 - epsilon**2) * px
    f2 = np.pi / np.sqrt(1 - epsilon**2) * py
    f = f1 + 1j * f2
    a[0] = f / (g - 1j * np.pi)
    # gbar = -g
    # fbar = -f^* = -f1 + 1j * f2
    # abar[0] = fbar / (gbar + 1j * np.pi)
    # for abar[-1], the x component f1 in the numerator gain an extra minus sign
    abar[-1] = f / (-g + 1j * np.pi)

    # Riccati equations
    def ricatti1(Delta1: complex, Delta2: complex, y0: complex) -> complex:
        ret = 1j * (
            epsilon * y0
            + 0.5 * (Delta1 - 1j * Delta2) * y0**2
            + 0.5 * (Delta1 + 1j * Delta2)
        )
        return ret / px

    def ricatti2(Delta1: complex, Delta2: complex, y0: complex) -> complex:
        ret = -1j * (
            epsilon * y0
            + 0.5 * (Delta1 + 1j * Delta2) * y0**2
            + 0.5 * (Delta1 - 1j * Delta2)
        )
        return ret / px

    def RK4(
        func: Callable[[complex, complex, complex], complex],
        Delta1_0: complex,
        Delta2_0: complex,
        Delta1_1: complex,
        Delta2_1: complex,
        Delta1_2: complex,
        Delta2_2: complex,
        y_0: complex,
        h: float,
    ) -> complex:
        k_1 = func(Delta1_0, Delta2_0, y_0)
        k_2 = func(Delta1_1, Delta2_1, y_0 + (h / 2) * k_1)
        k_3 = func(Delta1_1, Delta2_1, y_0 + (h / 2) * k_2)
        k_4 = func(Delta1_2, Delta2_2, y_0 + h * k_3)
        return y_0 + (h / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    for i in range(len(x) - 1):
        a[i + 1] = RK4(
            ricatti1,
            Delta1[i],
            Delta2[i],
            delta1[i],
            delta2[i],
            Delta1[i + 1],
            Delta2[i + 1],
            a[i],
            x[i + 1] - x[i],
        )
        idx = len(x) - i - 2
        abar[idx] = RK4(
            ricatti2,
            Delta1[idx + 1],
            Delta2[idx + 1],
            delta1[idx],
            delta2[idx],
            Delta1[idx],
            Delta2[idx],
            abar[idx + 1],
            x[idx] - x[idx + 1],
        )

    g = -1j * np.pi * (1 + a * abar) / (1 - a * abar)
    f = -1j * np.pi * 2 * a / (1 - a * abar)
    fbar = -1j * np.pi * (-2 * abar) / (1 - abar * a)
    f1 = (f - fbar) / 2
    f2 = (f + fbar) / 2 / 1j

    return x, f1, f2, g
