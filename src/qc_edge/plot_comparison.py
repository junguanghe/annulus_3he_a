import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import os

import matplotlib.pyplot as plt


def plot_gap_comparison():
    """Plot gap.txt and gap_1iter.txt together."""
    gap_file = "gap.txt"
    gap_1iter_file = "gap_1iter.txt"

    if not os.path.exists(gap_file):
        print(f"Warning: {gap_file} not found, skipping")
        return
    if not os.path.exists(gap_1iter_file):
        print(f"Warning: {gap_1iter_file} not found, skipping")
        return

    # Load data
    data = np.loadtxt(gap_file)
    xspan = data[:, 0]
    Delta1 = data[:, 1]
    Delta2 = data[:, 2]

    data_1iter = np.loadtxt(gap_1iter_file)
    xspan_1iter = data_1iter[:, 0]
    Delta1_1iter = data_1iter[:, 1]
    Delta2_1iter = data_1iter[:, 2]

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(xspan, Delta1, "k--", linewidth=1.5, label="Delta1 (converged)")
    plt.plot(xspan, Delta2, "k-", linewidth=1.5, label="Delta2 (converged)")
    plt.plot(
        xspan_1iter,
        Delta1_1iter,
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label="Delta1 (1 iter)",
    )
    plt.plot(
        xspan_1iter,
        Delta2_1iter,
        "r-",
        linewidth=1.5,
        alpha=0.7,
        label="Delta2 (1 iter)",
    )
    plt.title("Edge gap profile comparison")
    plt.xlabel(r"$x/\xi_\Delta$")
    plt.ylabel(r"$\Delta(x,T)/\Delta_0(T)$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_filename = "gap_comparison.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_filename}")
    plt.close()


def plot_jy_comparison():
    """Plot jy_vs_x.txt and jy_vs_x_1iter.txt together."""
    jy_file = "jy_vs_x.txt"
    jy_1iter_file = "jy_vs_x_1iter.txt"

    if not os.path.exists(jy_file):
        print(f"Warning: {jy_file} not found, skipping")
        return
    if not os.path.exists(jy_1iter_file):
        print(f"Warning: {jy_1iter_file} not found, skipping")
        return

    # Load data
    data = np.loadtxt(jy_file)
    xspan = data[:, 0]
    jy = data[:, 1]

    data_1iter = np.loadtxt(jy_1iter_file)
    xspan_1iter = data_1iter[:, 0]
    jy_1iter = data_1iter[:, 1]

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(xspan, jy, "k-", linewidth=1.5, label="jy (converged)")
    plt.plot(
        xspan_1iter, jy_1iter, "r-", linewidth=1.5, alpha=0.7, label="jy (1 iter)"
    )
    plt.title("jy vs x comparison")
    plt.xlabel(r"$x/\xi_\Delta$")
    plt.ylabel(r"$j_y/j_0$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_filename = "jy_comparison.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    plot_gap_comparison()
    plot_jy_comparison()

