import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import Slider
from utils import BLUE, RED
from scipy.stats import norm

# Initial Values
phi1 = 0.5
phi2 = 0
sigma = 1
t = 100

# Simulate Series
shocks = norm.rvs(loc=0, scale=sigma, size=t*2)
x = np.zeros(t * 2)
for i in range(2*t - 2):
    x[i + 2] = phi1 * x[i + 1] + phi2 * x[i] + shocks[i + 2]

x = x[-t:]


# Theoretical ACF
rho = np.zeros(30)
rho[0] = phi1 / (1 - phi2)
rho[1] = (phi1**2) / (1 - phi2) + phi2

for i in range(len(rho) - 2):
    rho[i + 2] = phi1 * rho[i + 1] + phi2 * rho[i]

# =================
# ===== Chart =====
# =================
size = 7
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# Series
ax_series = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
c_series, = ax_series.plot(x, color=BLUE)
ax_series.set_title("Simulated Series")
ax_series.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_series.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_series.tick_params(rotation=90, axis="x")

# ACF
ax_acf = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
c_acf = ax_acf.bar(list(range(1, len(rho) + 1)), rho, color=BLUE)
ax_acf.set_title("Simulated Series")
ax_acf.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_acf.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_acf.tick_params(rotation=90, axis="x")
ax_acf.set_ylim(-1, 1)


# Parameter Space
ax_p1p2 = plt.subplot2grid((6, 2), (0, 1), rowspan=6)

or_phi1 = np.arange(start=-2, stop=2, step=0.05)
or_phi2 = - 0.25 * or_phi1**2

ax_p1p2.set_title("Stationarity Condition")
c_p1p2, = ax_p1p2.plot(phi1, phi2, color=BLUE, ls=None, marker='d')
ax_p1p2.axvline(0, color="black", lw=0.5)
ax_p1p2.axhline(0, color="black", lw=0.5)

ax_p1p2.axline((0, 1), (1, 0), color=RED)
ax_p1p2.axline((-1, 0), (0, 1), color=RED)
ax_p1p2.axline((-1, -1), (1, -1), color=RED)
ax_p1p2.plot(or_phi1, or_phi2, color=RED, ls="--")

ax_p1p2.set_xlim(-2.2, 2.2)
ax_p1p2.set_ylim(-1.2, 1.2)
ax_p1p2.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_p1p2.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_p1p2.tick_params(rotation=90, axis="x")


# Sliders
ax_phi1 = plt.subplot2grid((6, 2), (4, 0))
s_phi1 = Slider(
    ax=ax_phi1,
    label="$\phi_{1}$",
    valmin=-2.2,
    valmax=2.2,
    valinit=phi1,
    valstep=0.05,
)

ax_phi2 = plt.subplot2grid((6, 2), (5, 0))
s_phi2 = Slider(
    ax=ax_phi2,
    label="$\phi_{2}$",
    valmin=-2.2,
    valmax=2.2,
    valinit=phi2,
    valstep=0.05,
)


def update(val):
    p1 = s_phi1.val
    p2 = s_phi2.val

    y = np.zeros(t * 2)
    for ii in range(2 * t - 2):
        y[ii + 2] = p1 * y[ii + 1] + p2 * y[ii] + shocks[ii + 2]

    y = y[-t:]

    # Theoretical ACF
    r = np.zeros(30)
    r[0] = p1 / (1 - p2)
    r[1] = (p1 ** 2) / (1 - p2) + p2

    for ii in range(len(r) - 2):
        r[ii + 2] = p1 * r[ii + 1] + p2 * r[ii]

    c_series.set_ydata(y)
    c_p1p2.set_xdata([p1])
    c_p1p2.set_ydata([p2])

    for bar, new_y in zip(c_acf, r):
        bar.set_height(new_y)

    fig.canvas.draw_idle()

s_phi1.on_changed(update)
s_phi2.on_changed(update)

plt.tight_layout()
plt.show()
