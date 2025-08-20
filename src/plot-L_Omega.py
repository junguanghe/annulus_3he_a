import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.family": "cm"})
# plt.rcParams.update({"font.size": 15})

Omega = np.linspace(-0.5, 1.5, 1000)
Lp = np.floor(Omega) + 1
Lm = Lp * 4 / 2.4 - 1
Lp = Lp * 4 / 2.4 + 1
Omega = (Omega + 0.5) * 2
plt.scatter(Omega, Lp, marker=".")
plt.scatter(Omega, Lm, marker=".")
plt.xlabel(r"$\Omega/\Omega_1$")
plt.ylabel(r"$L/L_0$")
plt.legend([r"$p+ip$", r"$p-ip$"])

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)

plt.grid(True)
plt.tight_layout()
plt.show()
fig.savefig("L_m.pdf")
