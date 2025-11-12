"""Numerical evaluation of the Josephson current for opposite-chirality p-wave leads."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.constants import pi
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from pwavepropagator_opposite import pwave_propagator_opposite


def _as_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Angle arrays must be one-dimensional.")
    return arr


def pwave_jsphsn_current_opposite(
    t: float,
    delta: float,
    thetaspan: Sequence[float],
    *,
    delta0: float = 1.7639,
    max_matsubara: int = 500,
    thetapspan: Sequence[float] | None = None,
    tolerance: float = 5e-4,
    show_progress: bool = True,
    gap_file: str | Path | None = None,
) -> Tuple[np.ndarray, int]:
    """Compute the Josephson current for opposite chirality p-wave superconductors.

    Parameters
    ----------
    t : float
        Reduced temperature (in units of ``T/T_c``).
    delta : float
        Bulk gap ratio for the chosen temperature.
    thetaspan : sequence of float
        Phase differences across the junction (radians) over which the current is evaluated.
    delta0 : float, optional
        Zero-temperature bulk gap ratio. Defaults to ``1.7639``.
    max_matsubara : int, optional
        Upper bound on the Matsubara index ``n``. Defaults to ``500``.
    thetapspan : sequence of float, optional
        Quasiclassical trajectory angles. Defaults to ``np.linspace(-pi/2, pi/2, 51)``.
    tolerance : float, optional
        Convergence criterion for the current (maximum absolute change). Defaults to ``5e-4``.
    show_progress : bool, optional
        Display a progress bar over Matsubara frequencies. Defaults to ``True``.
    gap_file : str or Path, optional
        Path to the gap profile text file. Defaults to ``gap.txt`` in the current directory.

    Returns
    -------
    tuple
        ``(I, n_converged)`` where ``I`` is the Josephson current as a numpy array and
        ``n_converged`` is the Matsubara index at which convergence was reached.
    """

    theta_values = _as_array(thetaspan)
    if theta_values.size == 0:
        raise ValueError("'thetaspan' must contain at least one angle.")

    if thetapspan is None:
        thetap_values = np.linspace(-math.pi / 2.0, math.pi / 2.0, 51)
    else:
        thetap_values = _as_array(thetapspan)
        if thetap_values.size == 0:
            raise ValueError(
                "'thetapspan' must contain at least one angle if provided."
            )

    prefactor = 16.0 * t / (delta0 * pi**2)
    cos_weights = np.cos(thetap_values)

    current = np.zeros_like(theta_values, dtype=float)
    gap_file = gap_file if gap_file is not None else "gap.txt"

    matsubara_iter = range(max_matsubara + 1)
    iterator = tqdm(matsubara_iter, desc="Matsubara n", disable=not show_progress)

    for n in iterator:
        epsilon = 1j * (2 * n + 1) * math.pi * t / delta
        if show_progress:
            iterator.set_postfix({"n": n, "epsilon": f"{epsilon.imag:.4f}j"})

        g_values = np.zeros((thetap_values.size, theta_values.size), dtype=complex)
        thetap_iter = tqdm(
            enumerate(thetap_values),
            total=thetap_values.size,
            desc=f"  thetap (n={n})",
            leave=False,
            disable=not show_progress,
        )
        for idx_thetap, thetap in thetap_iter:
            for idx_theta, theta in enumerate(theta_values):
                _, _, _, g_path = pwave_propagator_opposite(
                    thetap,
                    theta,
                    epsilon,
                    gap_file=gap_file,
                )
                g_values[idx_thetap, idx_theta] = g_path[g_path.size // 2]

        g_integrated = np.trapz(cos_weights[:, None] * g_values, thetap_values, axis=0)
        updated_current = current + prefactor * np.real(g_integrated)

        max_change = np.max(np.abs(updated_current - current))
        if show_progress:
            iterator.set_postfix({"n": n, "max_change": f"{max_change:.6f}"})
        if max_change < tolerance:
            current = updated_current
            iterator.close()
            if show_progress:
                print(
                    f"\n✓ Converged at n={n} (max_change={max_change:.6e} < {tolerance})"
                )
            return current, n

        current = updated_current

    iterator.close()
    return current, max_matsubara


__all__ = ["pwave_jsphsn_current_opposite"]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Test parameters
    t = 0.1
    delta = 1.7639
    thetaspan = np.linspace(0, 2 * np.pi, 101)

    print("Computing Josephson current for opposite chirality...")
    I, n_converged = pwave_jsphsn_current_opposite(
        t=t,
        delta=delta,
        thetaspan=thetaspan,
        max_matsubara=50,  # Reduced for faster testing
        show_progress=True,
    )

    print(f"\nConverged at Matsubara n={n_converged}")
    print(f"Current range: [{I.min():.6f}, {I.max():.6f}]")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(thetaspan / np.pi, I, "k-", linewidth=1.5)
    plt.xlabel(r"$\theta/\pi$")
    plt.ylabel(r"$I_s/I_{C0}$")
    plt.title(f"Josephson current (T={t}T_c, opposite chirality)")
    plt.grid(True)
    plt.tight_layout()

    output_file = "josephson_current_opposite.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")
    plt.close()
