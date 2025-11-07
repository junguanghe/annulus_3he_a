import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d


def propagator(thetap, epsilon):
    """
    Calculate the Green's function of a chiral p-wave superconductor near a wall.
    
    Parameters:
    -----------
    thetap : float
        Angle parameter
    epsilon : complex
        Energy in unit of bulk gap Delta_0, a typical value is 1j*(2*n+1)*pi*0.1/1.7639
    
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
    gap = loadmat('gap.mat')
    DDelta1 = gap['Delta1'].flatten()
    DDelta2 = gap['Delta2'].flatten()
    xspan = gap['xspan'].flatten()
    
    px = np.cos(thetap)
    py = np.sin(thetap)
    x = np.concatenate([xspan, -np.flip(xspan[:-1])])  # x coordinate of the classical trajectory
    
    if abs(thetap - np.pi/2) < 1e-10 or abs(thetap + np.pi/2) < 1e-10:
        f1 = np.zeros(len(x))
        f2 = np.pi * py / np.sqrt(1 - epsilon**2) * np.ones(len(x))
        g = -epsilon * np.pi / np.sqrt(1 - epsilon**2) * np.ones(len(x))
        return x, f1, f2, g
    
    t = x / px  # coordinate along a classical trajectory
    DDelta1 = np.concatenate([px * DDelta1, -px * np.flip(DDelta1[:-1])])  # encode a '-' from -px after bouncing off the wall
    DDelta2 = np.concatenate([py * DDelta2, py * np.flip(DDelta2[:-1])])
    
    # step size no bigger than 0.1xi
    t1 = np.arange(t[0] + 0.1, t[-1] - 0.1 + 0.1, 0.1)
    T = np.sort(np.concatenate([t, t1]))
    I = np.argsort(np.concatenate([t, t1]))
    II = np.argsort(I)
    
    # Interpolation
    interp_Delta1 = interp1d(t, DDelta1, kind='cubic', fill_value='extrapolate')
    interp_Delta2 = interp1d(t, DDelta2, kind='cubic', fill_value='extrapolate')
    Delta1 = interp_Delta1(T)
    Delta2 = interp_Delta2(T)
    
    Tq = (T[:-1] + T[1:]) / 2  # half step (used in 4th-order Runge Kutta method)
    delta1 = interp_Delta1(Tq)
    delta2 = interp_Delta2(Tq)
    
    a = np.zeros(len(T), dtype=complex)
    abar = np.zeros(len(T), dtype=complex)
    
    g = -np.pi * epsilon / np.sqrt(1 - epsilon**2)
    f1 = np.pi / np.sqrt(1 - epsilon**2) * px
    f2 = np.pi / np.sqrt(1 - epsilon**2) * py
    a[0] = -(f1 + 1j * f2) / (1j * np.pi - g)
    abar[-1] = -(-f1 - 1j * f2) / (1j * np.pi - g)  # initial condition for backward direction
    
    # Riccati equations
    def ricatti1(Delta1, Delta2, y0):
        return 1j * (epsilon * y0 + 0.5 * (Delta1 - 1j * Delta2) * y0**2 + 0.5 * (Delta1 + 1j * Delta2))
    
    def ricatti2(Delta1, Delta2, y0):
        return -1j * (epsilon * y0 + 0.5 * (Delta1 + 1j * Delta2) * y0**2 + 0.5 * (Delta1 - 1j * Delta2))
    
    def RK4(func, Delta1_0, Delta2_0, Delta1_1, Delta2_1, Delta1_2, Delta2_2, y_0, h):
        k_1 = func(Delta1_0, Delta2_0, y_0)
        k_2 = func(Delta1_1, Delta2_1, y_0 + (h/2) * k_1)
        k_3 = func(Delta1_1, Delta2_1, y_0 + (h/2) * k_2)
        k_4 = func(Delta1_2, Delta2_2, y_0 + h * k_3)
        return y_0 + (h/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
    
    for i in range(len(T) - 1):
        a[i+1] = RK4(ricatti1, Delta1[i], Delta2[i], delta1[i], delta2[i], 
                     Delta1[i+1], Delta2[i+1], a[i], T[i+1] - T[i])
        # Backward integration: MATLAB i goes from 1 to length(T)-1
        # Python i goes from 0 to len(T)-2
        # MATLAB end+1-i when i=1 → end = len(T)-1 in Python
        # MATLAB end-i when i=1 → end-1 = len(T)-2 in Python
        # So: end+1-i → len(T)-(i+1), end-i → len(T)-(i+2)
        matlab_i = i + 1  # Convert Python i to MATLAB i
        idx = len(T) - 1 - matlab_i  # abar[end-i] in MATLAB
        abar[idx] = RK4(ricatti2, Delta1[len(T) - matlab_i], Delta2[len(T) - matlab_i], 
                       delta1[idx], delta2[idx], 
                       Delta1[idx], Delta2[idx], 
                       abar[len(T) - matlab_i], T[idx] - T[len(T) - matlab_i])
    
    g = -1j * np.pi * (1 + a * abar) / (1 - a * abar)
    f = -1j * np.pi * 2 * a / (1 - a * abar)
    fbar = -1j * np.pi * (-2 * abar) / (1 - abar * a)
    f1 = (f - fbar) / 2
    f2 = (f + fbar) / 2 / 1j
    
    # Extract original x points
    num_x = len(x)
    g = g[II[:num_x]]
    f1 = f1[II[:num_x]]
    f2 = f2[II[:num_x]]
    
    return x, f1, f2, g

