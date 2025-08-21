import matplotlib.pyplot as plt
import numpy as np

a = np.load("dw_vs_chiral.npz")
D_dw_chiral = a[a.files[0]]
dtheta_dw_chiral = a[a.files[1]]

a = np.load("polar_vs_chiral.npz")
D_polar_chiral = a[a.files[0]]
dtheta_polar_chiral = a[a.files[1]]

a = np.load("polar_vs_dw.npz")
D_polar_dw = a[a.files[0]]
dtheta_polar_dw = a[a.files[1]]

plt.rcParams.update({"text.usetex": True, "font.family": "cm"})
# plt.plot(np.concatenate([D_polar_dw[::-1],D_dw_chiral]), np.concatenate([dtheta_polar_dw[::-1],dtheta_dw_chiral]), '-k', lw=2)
# plt.plot(D_polar_chiral, dtheta_polar_chiral, '-k', lw=2)
plt.plot(
    np.concatenate([D_polar_dw[::-1], D_dw_chiral]),
    np.concatenate([dtheta_polar_dw[::-1], dtheta_dw_chiral]),
    "-k",
    lw=2,
)
plt.plot(
    D_polar_chiral, dtheta_polar_chiral, "-k", lw=2
)  # times sqrt(2) to change the y-axis unit into vs/vc, where vc=1/(sqrt(2)xi)
plt.xlim(0, 30)
plt.ylim(0, 0.58)
plt.text(0.5, 0.25, "Polar")
plt.text(10, 0.05, "Chiral")
plt.text(10, 0.35, r"Axial DW")
plt.text(3, 0.1, r"$D_c\rightarrow$")
plt.text(20, 0.16, r"$v_T\downarrow$")
plt.xlabel(r"D/$\xi$")
plt.ylabel(r"$v_s/v_c$")
# plt.ylabel(r'$n/(R/\xi)$')

# ax = plt.gca()
# pos1 = ax.get_position() # get the original position
# pos2 = [pos1.x0 + 0.03, pos1.y0 + 0.05,  pos1.width, pos1.height]
# ax.set_position(pos2)

fig = plt.gcf()
fig.set_size_inches(4, 3)

plt.tight_layout()
plt.grid(True)
plt.savefig("phase_diagram.pdf")
