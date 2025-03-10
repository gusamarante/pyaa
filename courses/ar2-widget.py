import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import Slider
from utils import BLUE, RED
from scipy.stats import norm

# Initial Values
phi1 = 1.3
phi2 = -0.4
sigma = 1
t = 100

shocks = norm.rvs(loc=0, scale=sigma, size=t*2)

x = np.zeros(t * 2)


for i in range(2*t - 2):
    x[i + 2] = phi1 * x[i + 1] + phi2 * x[i] + shocks[i + 2]

x = x[-t:]



# =================
# ===== Chart =====
# =================
size = 7
fig = plt.figure(figsize=(size * (16 / 7.3), size))

# Series
ax_series = plt.subplot2grid((6, 2), (0, 0), rowspan=3)
c_series, = ax_series.plot(x, color=BLUE)
ax_series.set_title("Simulated Series")
ax_series.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_series.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_series.tick_params(rotation=90, axis="x")

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
ax_phi1 = plt.subplot2grid((6, 2), (3, 0))
s_phi1 = Slider(
    ax=ax_phi1,
    label="$\phi_{1}$",
    valmin=-2.2,
    valmax=2.2,
    valinit=phi1,
    valstep=0.05,
)

ax_phi2 = plt.subplot2grid((6, 2), (4, 0))
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

    c_series.set_ydata(y)
    c_p1p2.set_xdata([p1])
    c_p1p2.set_ydata([p2])
    fig.canvas.draw_idle()

s_phi1.on_changed(update)
s_phi2.on_changed(update)

plt.tight_layout()
plt.show()
