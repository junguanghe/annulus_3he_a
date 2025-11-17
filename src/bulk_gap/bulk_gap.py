import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def swavebulkgap():
    n = np.arange(10000)
    tspan = np.linspace(0.001, 1, 1000)
    delta = np.zeros_like(tspan)
    for i in tqdm(range(len(tspan))):
        t = tspan[i]
        d = np.array([0, 3.6])
        while np.abs(d[1] - d[0]) > 0.00001:
            newd = np.mean(d)
            diff = np.sum(
                1 / np.sqrt((n + 0.5) ** 2 + (newd / 2 / np.pi / t) ** 2)
                - 1 / (n + 0.5)
            ) - np.log(t)
            if diff > 0:
                d[0] = newd
            else:
                d[1] = newd
        delta[i] = np.mean(d)
    plt.plot(tspan, delta, "-", lw=2)
    plt.grid()
    plt.xlabel(r"$T/T_c$")
    plt.ylabel(r"$\Delta/T_c$")
    plt.savefig("bulk_gap.pdf")
    plt.show()

    # save delta and tspan to txt file
    np.savetxt("bulk_gap.txt", np.column_stack((tspan, delta)))
    return delta, tspan


delta, tspan = swavebulkgap()
