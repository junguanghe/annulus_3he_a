"""Unit tests for propagator_scipy.py"""

import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pytest
from propagator_scipy import propagator


def test_propagator_basic():
    """Test propagator function with gap.txt input, t=0.1 and delta=1.7639."""
    # Load gap.txt
    gap_file = os.path.join(os.path.dirname(__file__), "gap.txt")
    if not os.path.exists(gap_file):
        pytest.skip(f"gap.txt not found at {gap_file}")

    data = np.loadtxt(gap_file)
    xspan = data[:, 0]
    Delta1 = data[:, 1]
    Delta2 = data[:, 2]

    # Test parameters
    t = 0.1
    delta = 1.7639
    thetap = 0.1  # Angle parameter

    # Calculate epsilon for first Matsubara frequency (n=0)
    n = 0
    en = (2 * n + 1) * np.pi * t / delta
    epsilon = 1j * en

    # Call the propagator
    x, f1, f2, g = propagator(thetap, epsilon, xspan, Delta1, Delta2)

    # Verify output shapes
    assert (
        x.shape == f1.shape == f2.shape == g.shape
    ), "All output arrays should have the same shape"

    # Verify output types
    assert isinstance(x, np.ndarray), "x should be a numpy array"
    assert isinstance(f1, np.ndarray), "f1 should be a numpy array"
    assert isinstance(f2, np.ndarray), "f2 should be a numpy array"
    assert isinstance(g, np.ndarray), "g should be a numpy array"

    # Verify f1, f2, g are complex arrays
    assert np.iscomplexobj(f1), "f1 should be complex"
    assert np.iscomplexobj(f2), "f2 should be complex"
    assert np.iscomplexobj(g), "g should be complex"

    # Verify x is real
    assert np.isrealobj(x), "x should be real"

    # Verify output length is reasonable (should be 2*len(xspan) - 1)
    expected_len = 2 * len(xspan) - 1
    assert (
        len(x) == expected_len
    ), f"Expected output length {expected_len}, got {len(x)}"

    # Verify no NaN or Inf values
    assert not np.any(np.isnan(x)), "x should not contain NaN"
    assert not np.any(np.isnan(f1)), "f1 should not contain NaN"
    assert not np.any(np.isnan(f2)), "f2 should not contain NaN"
    assert not np.any(np.isnan(g)), "g should not contain NaN"

    assert not np.any(np.isinf(x)), "x should not contain Inf"
    assert not np.any(np.isinf(f1)), "f1 should not contain Inf"
    assert not np.any(np.isinf(f2)), "f2 should not contain Inf"
    assert not np.any(np.isinf(g)), "g should not contain Inf"

    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Propagator Scipy Test Results (t={t}, delta={delta}, thetap={thetap:.2f}, n={n})",
        fontsize=14,
    )

    # Plot f1 (real and imaginary parts)
    ax = axes[0, 0]
    ax.plot(x, np.real(f1), "b-", label="Re(f1)", linewidth=1.5)
    ax.plot(x, np.imag(f1), "r--", label="Im(f1)", linewidth=1.5)
    ax.set_xlabel(r"$x/\xi_\Delta$")
    ax.set_ylabel("f1")
    ax.set_title("f1 component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot f2 (real and imaginary parts)
    ax = axes[0, 1]
    ax.plot(x, np.real(f2), "b-", label="Re(f2)", linewidth=1.5)
    ax.plot(x, np.imag(f2), "r--", label="Im(f2)", linewidth=1.5)
    ax.set_xlabel(r"$x/\xi_\Delta$")
    ax.set_ylabel("f2")
    ax.set_title("f2 component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot g (real and imaginary parts)
    ax = axes[1, 0]
    ax.plot(x, np.real(g), "b-", label="Re(g)", linewidth=1.5)
    ax.plot(x, np.imag(g), "r--", label="Im(g)", linewidth=1.5)
    ax.set_xlabel(r"$x/\xi_\Delta$")
    ax.set_ylabel("g")
    ax.set_title("g component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot magnitudes
    ax = axes[1, 1]
    ax.plot(x, np.abs(f1), "b-", label="|f1|", linewidth=1.5)
    ax.plot(x, np.abs(f2), "g-", label="|f2|", linewidth=1.5)
    ax.plot(x, np.abs(g), "r-", label="|g|", linewidth=1.5)
    ax.set_xlabel(r"$x/\xi_\Delta$")
    ax.set_ylabel("Magnitude")
    ax.set_title("Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(
        os.path.dirname(__file__), "test_propagator_scipy_results.png"
    )
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {plot_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
