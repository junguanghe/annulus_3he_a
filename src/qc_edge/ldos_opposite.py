"""Compute and plot the LDOS for opposite-chirality point-contact junctions."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent

try:
    from pwavepropagator_opposite import pwave_propagator_opposite
except ModuleNotFoundError:  # pragma: no cover - defensive path handling
    sys.path.insert(0, str(MODULE_DIR))
    from pwavepropagator_opposite import pwave_propagator_opposite


def _ensure_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"'{name}' must be a 1D array with at least one element.")
    return arr


def ldos_opposite(
    theta: float,
    *,
    energy_range: tuple[float, float] = (-1.5, 1.5),
    energy_points: int = 301,
    py_points: int = 13,
    broadening: float = 0.025,
    temperature: float = 0.1,
    theta_py: Sequence[float] | None = None,
    gap_mat_path: str | Path | None = None,
    output_path: str | Path | None = "ldos_opposite.png",
    show_progress: bool = True,
    show: bool = False,
    interactive_backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the LDOS and save a plot showing forward and time-reversed contributions."""

    if show:
        try:
            if interactive_backend is not None:
                plt.switch_backend(interactive_backend)
            else:
                plt.switch_backend("QtAgg")
        except Exception:  # pragma: no cover - backend availability differs per env
            pass

    epsilon = np.linspace(energy_range[0], energy_range[1], energy_points)

    if theta_py is None:
        py = np.linspace(-1.0, 1.0, py_points)[1:-1]
    else:
        py = _ensure_array(theta_py, "theta_py")

    thetap = np.arcsin(np.clip(py, -1.0, 1.0))
    tr_thetap = thetap + math.pi

    g = np.zeros((thetap.size, epsilon.size), dtype=complex)
    tr_g = np.zeros_like(g)

    angle_pairs = list(zip(thetap, tr_thetap))
    angle_iterator = enumerate(angle_pairs)
    if show_progress:
        angle_iterator = tqdm(
            list(angle_iterator),
            desc="Angles",
            unit="angle",
            leave=False,
        )

    for idx, (th, tr) in angle_iterator:
        if show_progress:
            energy_iterator = tqdm(
                enumerate(epsilon),
                desc=f"Energy (idx={idx})",
                unit="eps",
                leave=False,
            )
        else:
            energy_iterator = enumerate(epsilon)

        for jdx, eps in energy_iterator:
            _, _, _, propagator_g = pwave_propagator_opposite(
                th,
                theta,
                eps + 1j * broadening,
                gap_mat_path=gap_mat_path,
            )
            _, _, _, propagator_tr = pwave_propagator_opposite(
                tr,
                theta,
                eps + 1j * broadening,
                gap_mat_path=gap_mat_path,
            )

            mid = propagator_g.size // 2
            g[idx, jdx] = propagator_g[mid]
            tr_g[idx, jdx] = propagator_tr[mid]

    N = -1.0 / math.pi * np.imag(g)
    TRN = -1.0 / math.pi * np.imag(tr_g)
    spectral_current = np.sqrt(1.0 - py[:, None] ** 2) * (N - TRN)

    epsilon_grid, py_grid = np.meshgrid(epsilon, py, indexing="xy")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis")
    ymin, ymax = py.min(), py.max()
    denom = ymax - ymin if ymax > ymin else 1.0

    for idx, py_val in enumerate(py):
        color = cmap((py_val - ymin) / denom)
        ax.plot(
            epsilon,
            np.full_like(epsilon, py_val),
            spectral_current[idx],
            color=color,
            linewidth=1.5,
        )

    max_abs = np.max(np.abs(spectral_current))
    if temperature > 0:
        fermi_profile = 1.0 / (np.exp(epsilon / temperature) + 1.0)
        fermi_profile *= max_abs / 4.0
        offset_py = py[-1] + 0.02
        ax.plot(
            epsilon,
            np.full_like(epsilon, offset_py),
            fermi_profile,
            color="blue",
            linewidth=2,
            label="Fermi window",
        )

    ax.set_xlabel(r"$\varepsilon/|\Delta|$")
    ax.set_ylabel(r"$p_y/p_F$")
    ax.set_zlabel(r"$\sqrt{1-p_y^2}\,(\mathcal{N}-\mathcal{N}_{\mathrm{TR}})$")
    ax.set_title(
        rf"Spectral current density for opposite chirality ($\theta = {theta/np.pi:.2f}\pi$)"
    )
    ax.view_init(elev=20.0, azim=-45.0)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return epsilon, py, N, TRN, spectral_current


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for CLI usage."""

    parser = argparse.ArgumentParser(
        description="Compute and visualize the LDOS for opposite-chirality point-contact junctions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Phase difference in units of π (e.g., 0.5 -> 0.5π).",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=-1.5,
        help="Minimum reduced energy ε/|Δ|.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=1.5,
        help="Maximum reduced energy ε/|Δ|.",
    )
    parser.add_argument(
        "--energy-points",
        type=int,
        default=301,
        help="Number of energy samples.",
    )
    parser.add_argument(
        "--py-points",
        type=int,
        default=13,
        help="Number of transverse momentum samples (including endpoints).",
    )
    parser.add_argument(
        "--broadening",
        type=float,
        default=0.025,
        help="Imaginary part added to energy for numerical stability.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the Fermi window overlay (in units of Tc).",
    )
    parser.add_argument(
        "--gap-mat",
        type=str,
        default=None,
        help="Path to gap.mat (defaults to file in this directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ldos_opposite.png",
        help="Output image path (set to empty string to skip saving).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in an interactive window.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Matplotlib backend to use when showing the figure (e.g., QtAgg).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )

    args = parser.parse_args(argv)

    energy_range = (args.energy_min, args.energy_max)
    output_path: str | Path | None = args.output if args.output else None

    ldos_opposite(
        theta=args.theta * math.pi,
        energy_range=energy_range,
        energy_points=args.energy_points,
        py_points=args.py_points,
        broadening=args.broadening,
        temperature=args.temperature,
        gap_mat_path=args.gap_mat,
        output_path=output_path,
        show_progress=not args.no_progress,
        show=args.show,
        interactive_backend=args.backend,
    )


if __name__ == "__main__":
    main()

__all__ = ["ldos_opposite"]
