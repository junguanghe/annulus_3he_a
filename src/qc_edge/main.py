import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import os

import matplotlib.pyplot as plt
from gap_equation import gap_equation


def initialize_gap_txt(
    xspan: np.ndarray,
    gap_file: str = "gap.txt",
    single_iteration: bool = False,
) -> None:
    """
    Initialize gap.txt file with initial gap profile.

    Parameters:
    -----------
    xspan : ndarray
        x coordinate span
    gap_file : str
        Filename to save gap data to
    Delta1_init : ndarray, optional
        Initial Delta1 profile. If None, uses bulk value of 1.0
    Delta2_init : ndarray, optional
        Initial Delta2 profile. If None, uses tanh profile
    """
    Delta2_init = np.ones_like(xspan)
    if single_iteration:
        Delta1_init = np.ones_like(xspan)
    else:
        Delta1_init = np.tanh(xspan[-1] - xspan)

    data = np.column_stack([xspan, Delta1_init, Delta2_init])
    np.savetxt(gap_file, data, header="xspan Delta1 Delta2", comments="#", fmt="%.10e")


def main(
    t: float, delta: float, reset_gap: bool = False, single_iteration: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Self-consistently calculate the gap profile of chiral p-wave superconductor near a wall.

    Parameters:
    -----------
    t : float
        Temperature parameter (e.g., 0.1 for T=0.1*Tc)
    delta : float
        Bulk gap ratio. when t=0.1, delta=1.7639. when t=0.5, delta=1.688
    reset_gap : bool, optional
        If True, delete gap.txt and start from scratch. Default is False.
    single_iteration : bool, optional
        If True, only iterate once (N=1) and automatically reset gap.txt.
        Also saves files with _1iter suffix. Default is False.

    Returns:
    --------
    xspan : ndarray
        x coordinate span
    Delta1 : ndarray
        First component of gap profile
    Delta2 : ndarray
        Second component of gap profile
    """
    N = 1 if single_iteration else 100  # max iteration number

    # If single_iteration, automatically reset gap and use _1iter suffix
    if single_iteration:
        reset_gap = True

    # Set filenames based on single_iteration flag
    if single_iteration:
        gap_file = "gap_1iter.txt"
        jy_file = "jy_vs_x_1iter.txt"
        gap_plot_file = "edge_gap_profile_1iter.png"
        jy_plot_file = "jy_vs_x_1iter.png"
    else:
        gap_file = "gap.txt"
        jy_file = "jy_vs_x.txt"
        gap_plot_file = "edge_gap_profile.png"
        jy_plot_file = "jy_vs_x.png"

    # Delete gap file if reset_gap is True
    if reset_gap and os.path.exists(gap_file):
        os.remove(gap_file)
        print(f"Deleted {gap_file}, starting from scratch")

    # Load or initialize gap profile
    if not os.path.exists(gap_file):
        # Create a reasonable initial xspan
        xspan = np.linspace(-15, 0, 100)  # Adjust range and resolution as needed
        initialize_gap_txt(xspan, gap_file, single_iteration)
        print(f"Initialized {gap_file} with default values")

    # Load initial gap profile
    data = np.loadtxt(gap_file)
    xspan = data[:, 0]
    Delta1 = data[:, 1]
    Delta2 = data[:, 2]

    for i in range(N):
        # Store previous values for convergence check
        Delta1_old = Delta1.copy()
        Delta2_old = Delta2.copy()

        # Calculate new gap profile
        xspan, Delta1, Delta2, jy, n = gap_equation(t, delta, xspan, Delta1, Delta2)
        print(f"iteration i={i+1}, Matsubara n={n}")

        # Check convergence: largest difference of Delta1 and Delta2 must be less than threshold
        max_diff_Delta1 = np.max(np.abs(Delta1 - Delta1_old))
        max_diff_Delta2 = np.max(np.abs(Delta2 - Delta2_old))
        print(f"max_diff_Delta1={max_diff_Delta1}, max_diff_Delta2={max_diff_Delta2}")
        if max_diff_Delta1 < 0.001 and max_diff_Delta2 < 0.001:
            print(f"Converged after {i+1} iterations")
            break

    # Save final gap profile
    data = np.column_stack([xspan, Delta1, Delta2])
    np.savetxt(gap_file, data, header="xspan Delta1 Delta2", comments="#", fmt="%.10e")

    # Final plot
    plt.figure(figsize=(8, 6))
    plt.plot(xspan, Delta1, "k--", linewidth=1.5, label="Delta1")
    plt.plot(xspan, Delta2, "k--", linewidth=1.5, label="Delta2")
    plt.title("Edge gap profile")
    plt.xlabel(r"$x/\xi_\Delta$")
    plt.ylabel(r"$\Delta(x,T)/\Delta_0(T)$")
    plt.legend([f"T={t}T_c"])
    plt.grid(True)
    plt.tight_layout()

    # Save plot to file
    plt.savefig(gap_plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {gap_plot_file}")
    plt.close()

    # Save jy and x to txt file
    jy_data = np.column_stack([xspan, jy])
    np.savetxt(jy_file, jy_data, header="xspan jy", comments="#", fmt="%.10e")
    print(f"jy and x data saved to {jy_file}")

    # make another plot for jy vs x
    plt.figure(figsize=(8, 6))
    plt.plot(xspan, jy, "k--", linewidth=1.5, label="jy")
    plt.title("jy vs x")
    plt.xlabel(r"$x/\xi_\Delta$")
    plt.ylabel(r"$j_y/j_0$")
    plt.legend([f"T={t}T_c"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(jy_plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {jy_plot_file}")
    plt.close()

    # also calculate int jy(x) dx
    angular_momentum = np.trapz(jy, xspan) * 2 * t / delta
    print(f"angular_momentum={angular_momentum} N hbar / 2")

    return xspan, Delta1, Delta2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate gap profile of chiral p-wave superconductor near a wall"
    )
    parser.add_argument(
        "--reset-gap", action="store_true", help="Delete gap.txt and start from scratch"
    )
    parser.add_argument(
        "--single-iteration", action="store_true", help="Only iterate once (set N=1)"
    )

    args = parser.parse_args()

    # Example usage
    t = 0.1
    delta = 1.7639
    xspan, Delta1, Delta2 = main(
        t, delta, reset_gap=args.reset_gap, single_iteration=args.single_iteration
    )
