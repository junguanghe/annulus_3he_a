import numpy as np
from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from gap_equation import gap_equation
import os


def initialize_gap_mat(xspan, Delta1_init=None, Delta2_init=None):
    """
    Initialize gap.mat file with initial gap profile.
    
    Parameters:
    -----------
    xspan : ndarray
        x coordinate span
    Delta1_init : ndarray, optional
        Initial Delta1 profile. If None, uses bulk value of 1.0
    Delta2_init : ndarray, optional
        Initial Delta2 profile. If None, uses tanh profile
    """
    if Delta1_init is None:
        Delta1_init = np.ones_like(xspan)
    if Delta2_init is None:
        Delta2_init = np.tanh(xspan[-1] - xspan)
    
    savemat('gap.mat', {
        'xspan': xspan,
        'Delta1': Delta1_init,
        'Delta2': Delta2_init
    })


def main(t, delta):
    """
    Self-consistently calculate the gap profile of chiral p-wave superconductor near a wall.
    
    Parameters:
    -----------
    t : float
        Temperature parameter (e.g., 0.1 for T=0.1*Tc)
    delta : float
        Bulk gap ratio. when t=0.1, delta=1.7639. when t=0.5, delta=1.688
    
    Returns:
    --------
    xspan : ndarray
        x coordinate span
    Delta1 : ndarray
        First component of gap profile
    Delta2 : ndarray
        Second component of gap profile
    """
    N = 100  # max iteration number
    
    # Initialize gap.mat if it doesn't exist
    if not os.path.exists('gap.mat'):
        # Create a reasonable initial xspan
        xspan = np.linspace(0, 10, 201)  # Adjust range and resolution as needed
        initialize_gap_mat(xspan)
        print("Initialized gap.mat with default values")
    
    for i in range(1, N + 1):
        gap = loadmat('gap.mat')
        xspan, Delta1, Delta2, n = gap_equation(t, delta)
        print(f'iteration i={i}, Matsubara n={n}')
        
        if i > 1:
            Delta2_old = gap['Delta2'].flatten()
            if abs(Delta2[-1] - Delta2_old[-1]) < 0.001:
                savemat('gap.mat', {
                    'xspan': xspan,
                    'Delta1': Delta1,
                    'Delta2': Delta2
                })
                break
        
        savemat('gap.mat', {
            'xspan': xspan,
            'Delta1': Delta1,
            'Delta2': Delta2
        })
    
    # Final plot
    plt.figure(figsize=(8, 6))
    plt.plot(xspan, Delta1, 'k--', linewidth=1.5, label='Delta1')
    plt.plot(xspan, Delta2, 'k--', linewidth=1.5, label='Delta2')
    plt.title('Edge gap profile')
    plt.xlabel(r'$x/\xi_\Delta$')
    plt.ylabel(r'$\Delta(x,T)/\Delta_0(T)$')
    plt.legend([f'T={t}T_c'])
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    plot_filename = 'edge_gap_profile.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {plot_filename}')
    plt.close()
    
    return xspan, Delta1, Delta2


if __name__ == '__main__':
    # Example usage
    t = 0.1
    delta = 1.7639
    xspan, Delta1, Delta2 = main(t, delta)

