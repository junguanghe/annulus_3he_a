import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize


def boundary_cond(eta1, eta2, BC):
    if BC:
        eta1[0] = (
            1j * (3 * eta2[1] - 1j * eta1[1]) / 4
        )  # boundary condition for r=0. also works for bulk
        eta1[-1] = eta1[-2]
        eta2[0] = (3 * eta2[1] - 1j * eta1[1]) / 4
        eta2[-1] = 0
    else:
        eta1[0] = (
            1j * (3 * eta2[1] - 1j * eta1[1]) / 4
        )  # boundary condition for r=0. also works for bulk
        eta1[-1] = 0
        eta2[0] = (3 * eta2[1] - 1j * eta1[1]) / 4
        eta2[-1] = 0
    return eta1, eta2


def GL(u, h, n, r, BC):
    eta1 = u[0, :] + u[1, :] * 1j
    eta2 = u[2, :] + u[3, :] * 1j

    eta1, eta2 = boundary_cond(eta1, eta2, BC)

    ees = eta1 * np.conj(eta1) + eta2 * np.conj(eta2)
    ee = eta1**2 + eta2**2

    de1 = np.gradient(eta1, h)
    de2 = np.gradient(eta2, h)

    G1 = (
        -eta1
        + 0.5 * ees * eta1
        + 0.25 * ee * np.conj(eta1)
        + (1 + 3 * n**2) * eta1 / r**2
        - 4 * 1j * n * eta2 / r**2
        - (de1 / r + 2 * 1j * n * de2 / r)
    )
    G2 = (
        -eta2
        + 0.5 * ees * eta2
        + 0.25 * ee * np.conj(eta2)
        + (3 + n**2) * eta2 / r**2
        + 4 * 1j * n * eta1 / r**2
        - (3 * de2 / r + 2 * 1j * n * de1 / r)
    )
    G1[1:-1] -= np.diff(eta1, n=2) / h**2
    G2[1:-1] -= 3 * np.diff(eta2, n=2) / h**2

    G1[0] = 0
    G1[-1] = 0
    G2[0] = 0
    G2[-1] = 0

    G = np.array([G1.real, G1.imag, G2.real, G2.imag])
    return G


def calc_j_f_L(eta1, eta2, dtheta, r, h):
    ees = eta1 * np.conj(eta1) + eta2 * np.conj(eta2)
    ee = eta1**2 + eta2**2

    de1 = np.gradient(eta1, h)
    de2 = np.gradient(eta2, h)

    d1 = 1j * dtheta / r
    dd1 = -(dtheta**2) / r**2

    f = (
        -ees
        + 0.25 * ees**2
        + 0.125 * ee * np.conj(ee)
        - ees * dd1
        + de1 * np.conj(de1)
        + 3 * de2 * np.conj(de2)
        - 2 * eta1 * np.conj(eta1) * dd1
        + d1
        * (
            eta1 * np.conj(de2)
            + eta2 * np.conj(de1)
            - np.conj(eta1) * de2
            - np.conj(eta2) * de1
        )
    )

    f += 2 / r * (
        3 * np.conj(eta2) * eta1 * d1
        - np.conj(eta1) * eta2 * d1
        + np.conj(eta2) * de2
        - np.conj(eta1) * de1
    ).real + 1 / r**2 * (
        ees + 2 * eta2 * np.conj(eta2)
    )  # extra terms besides the terms in Cartesian basis

    F = integrate.trapezoid(r[1:] * f[1:], dx=h) * 2 * np.pi

    j1_phi = (
        3 * np.conj(eta1) * eta1 * d1
        + np.conj(eta2) * eta2 * d1
        + 4 * np.conj(eta1) * eta2 / r
    )

    j1_r = np.conj(eta2) * de1 + np.conj(eta1) * de2

    j1 = (
        3 * np.conj(eta1) * eta1 * d1
        + np.conj(eta2) * eta2 * d1
        + np.conj(eta2) * de1
        + np.conj(eta1) * de2
        + 4 * np.conj(eta1) * eta2 / r
    )

    j2 = (
        3 * np.conj(eta2) * de2
        + np.conj(eta1) * de1
        + np.conj(eta1) * eta2 * d1
        + np.conj(eta2) * eta1 * d1
    )

    j1[0] = 0

    l = r * j1.imag

    L = integrate.trapezoid(r * l, dx=h) * 2 * np.pi  # in unit of

    L_phi = integrate.trapezoid(r**2 * j1_phi.imag, dx=h) * 2 * np.pi

    L_r = integrate.trapezoid(r**2 * j1_r.imag, dx=h) * 2 * np.pi

    return j1, j2, f, F, l, L, L_phi, L_r


def plot_results(eta1, eta2, j1, j2, f, r, dtheta):
    plt.figure()

    plt.subplot(411)
    plt.plot(r, np.abs(eta1), "-", lw=2, label=r"$\eta_\phi$")
    plt.plot(r, np.abs(eta2), "-", lw=2, label=r"$\eta_r$")
    plt.ylabel(r"$|\eta|/\eta_0$")
    plt.legend()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.title(r"$\partial_\phi\theta=$" + str(dtheta) + r", $\phi=0$")

    plt.subplot(412)
    plt.plot(r[1:-1], (np.angle(eta1[1:-1])) / np.pi, label=r"$\theta_\phi$")
    plt.plot(r[1:-1], (np.angle(eta2[1:-1])) / np.pi, label=r"$\theta_r$")
    plt.ylabel(r"$\theta/\pi$")
    plt.legend()
    frame2 = plt.gca()
    frame2.axes.xaxis.set_ticklabels([])

    plt.subplot(413)
    plt.plot(r, j1.imag, "-", lw=2, label=r"$j_\phi$")
    plt.plot(r, j2.imag, "-", lw=2, label=r"$j_r$")
    plt.ylabel(r"j /$j_0$")
    plt.legend()
    frame3 = plt.gca()
    frame3.axes.xaxis.set_ticklabels([])

    plt.subplot(414)
    plt.plot(r[1:], f[1:].real, "-", lw=2)
    plt.plot(r, np.zeros_like(r), "--k", lw=1)
    plt.ylabel(r"f/$f_0$")
    plt.xlabel(r"r/$\xi$")

    plt.show()
    return


def lz_r(nr, R0):
    ############################### Input
    # nr = 1001 # number of grid points along radial direction
    dtheta = 1  # d(theta)/d(phi) which should be an integer(winding number)
    h = 0.1  # step size
    # R0 = 1e-3 # inner radius of the annulus
    tol = 1e-10
    BC = True  # True = minimal pair-breaking, False = maximal pair-breaking

    ############################### calculate parameters
    D = (nr - 1) * h  # width of the annulus
    r = R0 + np.linspace(0, D, nr)  # radial coordinates r
    V1 = np.pi * R0**2
    V2 = np.pi * (R0 + D) ** 2

    ############################### initial guess
    eta0 = np.array(
        [1j * np.ones_like(r), np.tanh(D / 2 - np.abs(r - R0 - D / 2))], dtype=complex
    )  # chiral phase p+ip
    # eta0 = np.array([1j*np.tanh(r-R0-D/2), np.tanh(D/2-np.abs(r-R0-D/2))], dtype=complex)   # p-ip vs p+ip
    # eta0 = np.array([-1j*np.tanh(r-R0-D/2), np.tanh(D/2-np.abs(r-R0-D/2))], dtype=complex)   # p-ip vs p+ip

    ############################### solve GL
    u0 = np.array([eta0[0, :].real, eta0[0, :].imag, eta0[1, :].real, eta0[1, :].imag])
    sol = optimize.root(
        GL,
        u0,
        args=(h, dtheta, r, BC),
        method="krylov",
        options={"disp": False, "xatol": tol},
    )
    u = sol.x

    eta1 = u[0, :] + u[1, :] * 1j  # eta_phi
    eta2 = u[2, :] + u[3, :] * 1j  # eta_r

    ############################ set boundary values of order parameters
    eta1, eta2 = boundary_cond(eta1, eta2, BC)

    ############################ calculate mass current, free energy and angular momentum
    j1, j2, f, F, l, L, L_phi, L_r = calc_j_f_L(eta1, eta2, dtheta, r, h)
    # print("average Free Energy density=", (F-1*V1)/V2)
    # print("Angular momentum L =", L/V2)

    ############################ plot results
    # plot_results(eta1, eta2, j1, j2, f, r, dtheta)

    return L / V2, (F - V1) / V2


N = 301
nc = 30
R0 = 1e-3
n = np.linspace(nc, N, N + 1 - nc)
l = np.zeros(N + 1 - nc)
for ii in range(N - nc + 1):
    nr = int(n[ii])
    l[ii] = lz_r(nr, R0)[0]
    print(ii + nc, l[ii])

N1 = 30
l1 = np.zeros(N1)
R = np.logspace(0, 6, N1)
for ii in range(N1):
    R0 = R[ii]
    l1[ii] = lz_r(nr, R0)[0]
    print(ii, l1[ii])

x = np.append((n - 1) * 0.1, R + (N - 1) * 0.1)
y = np.append(l, l1)

plt.rcParams.update({"text.usetex": True, "font.family": "cm"})
plt.rcParams.update({"font.size": 15})

fig, ax = plt.subplots()
ax.plot(x, y, label="Chiral")
ax.plot(np.linspace(0, 0.1 * (nc - 1)), np.zeros(50), label="Normal State")
ax.set_xscale("log")
ax.set_xlabel(r"$R/\xi$")
ax.set_ylabel(r"$L(R)/2l_cV$")
ax.legend()
plt.tight_layout()
ax.grid(True)
fig.savefig("L_r.pdf")
