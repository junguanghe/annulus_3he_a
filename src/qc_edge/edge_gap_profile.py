import numpy as np
from scipy import optimize
from tqdm import tqdm

t = 0.01


def bulkG(n: int, t: float, d: float) -> tuple[float, float]:
    en = (2 * n + 1) * np.pi * t
    comm = np.pi / np.sqrt(en**2 + d**2)
    g = -1j * en * comm
    f = np.pi * d * comm
    return g, f


def eilenberger(
    arr: np.ndarray,  # shape: (4*N,) flattened
    d: np.ndarray,  # shape: (N,)
    n: int,
    t: float,
    dx: float,
    p: float,
) -> np.ndarray:
    N = len(d)
    arr = arr.reshape(4, N)
    gr = arr[0]
    gi = arr[1]
    f1 = arr[2]
    f2 = arr[3]
    g = gr + 1j * gi
    f = f1 + 1j * f2
    fconj = f1 - 1j * f2
    dg = np.gradient(g, dx)
    df = np.gradient(f, dx)
    en = (2 * n + 1) * np.pi * t
    px = np.cos(p)
    eq1 = 1j * en * g + d * fconj + 1j * px * dg
    eq2 = 1j * en * f + d * g + 1j * px * df
    eq1_r = eq1.real
    eq1_i = eq1.imag
    eq2_r = eq2.real
    eq2_i = eq2.imag
    return np.concatenate((eq1_r, eq1_i, eq2_r, eq2_i))


Nx = 1001
dx = 0.01
Np = 101
dp = np.pi / Np
ps = np.linspace(-np.pi / 2, np.pi / 2 - dp, Np) + dp / 2
tol = 1e-6

x = np.arange(0, Nx * dx, dx)
d1 = np.ones(Nx)
d2 = np.tanh(x[-1] - x)
od1 = np.zeros(Nx)
od2 = np.zeros(Nx)

while max(abs(od1 - d1)) > tol or max(abs(od2 - d2)) > tol:
    dd1 = np.concatenate((d1, -d1[-2::-1]))
    dd2 = np.concatenate((d2, d2[-2::-1]))
    n = 0
    while True:
        g0, f0 = bulkG(n, t, dd1[0])
        sols = []
        for p in tqdm(ps):
            d = dd1 * np.cos(p) + 1j * dd2 * np.sin(p)
            f10 = f0 * np.cos(p)
            f20 = f0 * np.sin(p)
            tmp = np.ones_like(d1) * f10
            tmp = np.concatenate((tmp, -tmp[:-1]))
            init_guess = np.concatenate(
                (
                    np.ones_like(dd1) * g0.real,
                    np.ones_like(dd1) * g0.imag,
                    tmp,
                    np.ones_like(dd1) * f20,
                )
            )
            sol = optimize.root(
                eilenberger,
                init_guess,
                args=(d, n, t, dx, p),
                method="krylov",
                options={"disp": True},
            )
            sols.append(sol.x)
        break
    break
