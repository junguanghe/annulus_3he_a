"""
Propagator implementation using scipy's solve_ivp ODE solver.

This is a modernized version of propagator.py that uses scipy.integrate.solve_ivp
instead of manual Runge-Kutta integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
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

    This implementation uses scipy's solve_ivp for ODE integration.

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

    a = np.zeros(len(x), dtype=complex)
    abar = np.zeros(len(x), dtype=complex)

    g = -np.pi * epsilon / np.sqrt(1 - epsilon**2)
    f1 = np.pi / np.sqrt(1 - epsilon**2) * px
    f2 = np.pi / np.sqrt(1 - epsilon**2) * py
    f = f1 + 1j * f2
    a0 = f / (g - 1j * np.pi)
    # gbar = -g
    # fbar = -f^* = -f1 + 1j * f2
    # abar[0] = fbar / (gbar + 1j * np.pi)
    # for abar[-1], the x component f1 in the numerator gain an extra minus sign
    abar_end = f / (-g + 1j * np.pi)

    # Convert complex initial conditions to real arrays [real, imag]
    a0_real = np.array([a0.real, a0.imag])
    abar_end_real = np.array([abar_end.real, abar_end.imag])

    # Riccati equations
    # Define Riccati equations as ODE functions for solve_ivp
    # solve_ivp works with real arrays, so we convert complex ODEs to real ODEs
    def ricatti1_ode(t: float, y: np.ndarray) -> np.ndarray:
        """
        Riccati equation 1: da/dx = 1j * (epsilon * a + 0.5 * (Delta1 - 1j*Delta2) * a^2 + 0.5 * (Delta1 + 1j*Delta2))
        Converted to real form: y = [Re(a), Im(a)]
        """
        Delta1_t = interp_Delta1(t)
        Delta2_t = interp_Delta2(t)
        a = y[0] + 1j * y[1]  # Convert to complex

        # Compute derivative
        da_dx = 1j * (
            epsilon * a
            + 0.5 * (Delta1_t - 1j * Delta2_t) * a**2
            + 0.5 * (Delta1_t + 1j * Delta2_t)
        )
        da_dx /= px

        # Convert back to real array
        return np.array([da_dx.real, da_dx.imag])

    def ricatti2_ode(t: float, y: np.ndarray) -> np.ndarray:
        """
        Riccati equation 2: dabar/dx = -1j * (epsilon * abar + 0.5 * (Delta1 + 1j*Delta2) * abar^2 + 0.5 * (Delta1 - 1j*Delta2))
        Note: This is integrated backwards, so we need to handle the sign.
        Converted to real form: y = [Re(abar), Im(abar)]
        """
        Delta1_t = interp_Delta1(t)
        Delta2_t = interp_Delta2(t)
        abar = y[0] + 1j * y[1]  # Convert to complex

        # Compute derivative (note the minus sign for backward integration)
        dabar_dx = -1j * (
            epsilon * abar
            + 0.5 * (Delta1_t + 1j * Delta2_t) * abar**2
            + 0.5 * (Delta1_t - 1j * Delta2_t)
        )
        dabar_dx /= px

        # Convert back to real array
        return np.array([dabar_dx.real, dabar_dx.imag])

    # Forward integration for a
    t_span_forward = (x[0], x[-1])
    t_eval_forward = x

    sol_forward = solve_ivp(
        ricatti1_ode,
        t_span_forward,
        a0_real,
        t_eval=t_eval_forward,
        method="RK45",  # Default adaptive Runge-Kutta method
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol_forward.success:
        raise RuntimeError(f"Forward integration failed: {sol_forward.message}")

    a = sol_forward.y[0, :] + 1j * sol_forward.y[1, :]

    # Backward integration for abar
    # For backward integration, we integrate from x[-1] to x[0]
    # by using negative time direction
    t_span_backward = (x[-1], x[0])
    t_eval_backward = np.flip(x)  # Evaluate at reversed x points

    sol_backward = solve_ivp(
        ricatti2_ode,
        t_span_backward,
        abar_end_real,
        t_eval=t_eval_backward,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol_backward.success:
        raise RuntimeError(f"Backward integration failed: {sol_backward.message}")

    # Reverse the solution to match the original x order
    abar = np.flip(sol_backward.y[0, :] + 1j * sol_backward.y[1, :])

    # Compute final Green's function components
    g = -1j * np.pi * (1 + a * abar) / (1 - a * abar)
    f = -1j * np.pi * 2 * a / (1 - a * abar)
    fbar = -1j * np.pi * (-2 * abar) / (1 - abar * a)
    f1 = (f - fbar) / 2
    f2 = (f + fbar) / 2 / 1j

    return x, f1, f2, g
