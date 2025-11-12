import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator


def _rk4_step(
    func,
    delta1_0: complex,
    delta2_0: complex,
    delta1_half: complex,
    delta2_half: complex,
    delta1_1: complex,
    delta2_1: complex,
    y0: complex,
    h: float,
):
    """Fourth-order Runge–Kutta step for the Riccati equations."""

    k1 = func(delta1_0, delta2_0, y0)
    k2 = func(delta1_half, delta2_half, y0 + 0.5 * h * k1)
    k3 = func(delta1_half, delta2_half, y0 + 0.5 * h * k2)
    k4 = func(delta1_1, delta2_1, y0 + h * k3)
    return y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def pwave_propagator_opposite(
    thetap: float,
    theta: float,
    epsilon: complex,
    *,
    gap_file: str | Path | None = None,
    min_step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the propagator for opposite-chirality p-wave superconductors at a point contact.

    Parameters
    ----------
    thetap : float
        Trajectory angle relative to the x-axis.
    theta : float
        Phase difference between the two superconductors.
    epsilon : complex
        Matsubara frequency (in units of the bulk gap).
    gap_file : str | Path | None, optional
        Path to gap profile text file. Defaults to ``gap.txt`` in the current directory.
    min_step : float, optional
        Maximum step size along the trajectory in coherence-length units.

    Returns
    -------
    tuple
        ``(x, f1, f2, g)`` arrays along the classical trajectory.
    """

    if gap_file is None:
        gap_file = Path(__file__).resolve().parent / "gap.txt"
    else:
        gap_file = Path(gap_file)

    data = np.loadtxt(gap_file)
    xspan = data[:, 0]
    delta1_raw = data[:, 1]
    delta2_raw = data[:, 2]

    px = float(np.cos(thetap))
    py = float(np.sin(thetap))

    # Classical trajectory positions (mirror reflected around the interface at x = 0).
    x = np.concatenate([xspan, -np.flip(xspan[:-1])])

    if np.isclose(px, 0.0, atol=1e-12):
        denom = np.sqrt(1.0 - epsilon**2)
        prefactor = np.pi / denom
        f1 = np.zeros_like(x, dtype=complex)
        f2 = -prefactor * py * np.ones_like(x, dtype=complex)
        g = -epsilon * prefactor * np.ones_like(x, dtype=complex)
        return x, f1, f2, g

    t_raw = x / px

    # Combine left and right order parameters with phase difference ``theta``.
    left_gap = np.exp(-0.5j * theta) * (delta1_raw * px - 1j * delta2_raw * py)
    right_gap = np.exp(0.5j * theta) * (delta1_raw * px + 1j * delta2_raw * py)

    bridge = 0.5 * (left_gap[-1] + right_gap[-1])
    delta_combined = np.concatenate([left_gap[:-1], [bridge], right_gap[:-1][::-1]])

    d_delta1_raw = np.real(delta_combined)
    d_delta2_raw = np.imag(delta_combined)

    sort_idx = np.argsort(t_raw)
    t = t_raw[sort_idx]
    d_delta1 = d_delta1_raw[sort_idx]
    d_delta2 = d_delta2_raw[sort_idx]

    # Build densified grid along the trajectory.
    num_segments = max(int(np.ceil(abs(t[-1] - t[0]) / min_step)), 1)
    t_dense = np.linspace(t[0], t[-1], num_segments + 1)

    if t_dense.size > 2:
        merged = np.concatenate([t, t_dense[1:-1]])
    else:
        merged = t.copy()

    T = np.sort(merged)

    # Interpolants
    interp_delta1 = PchipInterpolator(t, d_delta1)
    interp_delta2 = PchipInterpolator(t, d_delta2)

    Delta1 = interp_delta1(T)
    Delta2 = interp_delta2(T)

    T_mid = 0.5 * (T[:-1] + T[1:])
    delta1_mid = interp_delta1(T_mid)
    delta2_mid = interp_delta2(T_mid)

    a = np.zeros_like(T, dtype=complex)
    abar = np.zeros_like(T, dtype=complex)

    denom = np.sqrt(1.0 - epsilon**2)
    g_bulk = -np.pi * epsilon / denom

    left_bulk_delta = np.exp(-0.5j * theta) * (px - 1j * py)
    right_bulk_delta = np.exp(0.5j * theta) * (px + 1j * py)

    prefactor = np.pi / denom
    f1_left = prefactor * np.real(left_bulk_delta)
    f2_left = prefactor * np.imag(left_bulk_delta)
    f1_right = prefactor * np.real(right_bulk_delta)
    f2_right = prefactor * np.imag(right_bulk_delta)

    if px > 0:
        a[0] = -(f1_left + 1j * f2_left) / (1j * np.pi - g_bulk)
    else:
        abar[-1] = -(f1_left - 1j * f2_left) / (1j * np.pi - g_bulk)

    if px > 0:
        abar[-1] = -(f1_right - 1j * f2_right) / (1j * np.pi - g_bulk)
    else:
        a[0] = -(f1_right + 1j * f2_right) / (1j * np.pi - g_bulk)

    def ricatti1(delta1_val, delta2_val, y0):
        return 1j * (
            epsilon * y0
            + 0.5 * (delta1_val - 1j * delta2_val) * y0**2
            + 0.5 * (delta1_val + 1j * delta2_val)
        )

    def ricatti2(delta1_val, delta2_val, y0):
        return -1j * (
            epsilon * y0
            + 0.5 * (delta1_val + 1j * delta2_val) * y0**2
            + 0.5 * (delta1_val - 1j * delta2_val)
        )

    segment_count = T.size - 1

    for idx in range(segment_count):
        h = T[idx + 1] - T[idx]
        a[idx + 1] = _rk4_step(
            ricatti1,
            Delta1[idx],
            Delta2[idx],
            delta1_mid[idx],
            delta2_mid[idx],
            Delta1[idx + 1],
            Delta2[idx + 1],
            a[idx],
            h,
        )

    for idx in range(segment_count - 1, -1, -1):
        h = T[idx] - T[idx + 1]
        abar[idx] = _rk4_step(
            ricatti2,
            Delta1[idx + 1],
            Delta2[idx + 1],
            delta1_mid[idx],
            delta2_mid[idx],
            Delta1[idx],
            Delta2[idx],
            abar[idx + 1],
            h,
        )

    a = np.asarray(a)
    abar = np.asarray(abar)

    g = -1j * np.pi * (1.0 + a * abar) / (1.0 - a * abar)
    f = -1j * np.pi * 2.0 * a / (1.0 - a * abar)
    fbar = -1j * np.pi * (-2.0 * abar) / (1.0 - abar * a)
    f1 = 0.5 * (f - fbar)
    f2 = 0.5 / 1j * (f + fbar)

    original_positions = np.searchsorted(T, t)
    g_sorted = g[original_positions]
    f1_sorted = f1[original_positions]
    f2_sorted = f2[original_positions]

    unsort_idx = np.empty_like(sort_idx)
    unsort_idx[sort_idx] = np.arange(sort_idx.size)

    g_final = g_sorted[unsort_idx]
    f1_final = f1_sorted[unsort_idx]
    f2_final = f2_sorted[unsort_idx]

    return x, f1_final, f2_final, g_final


__all__ = ["pwave_propagator_opposite"]


