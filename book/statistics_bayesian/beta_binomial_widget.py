"""
Beta-Binomial conjugate with slider widget
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.stats import beta
from pylab import Slider

size = 4
n = 10

inc = 1 / 100  # step size for theta grid
theta_grid = np.arange(start=0, stop=1 + inc, step=inc)

def post(t, y, a, b):
    a1 = a + y
    b1 = n - y + b
    return beta.pdf(x=t, a=a1, b=b1)

def prior(t, a, b):
    return beta.pdf(x=t, a=a, b=b)


a_range = np.arange(0,11, 1)
b_range = np.arange(0,11, 1)
y_range = np.arange(0, n + 1, 1)



# =====================================================
# ===== Chart - Observed, Fitted and Risk-Neutral =====
# =====================================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))
fig.subplots_adjust(bottom=0.26)

ax = plt.subplot2grid((1, 1), (0, 0))
c_prior, = ax.plot(theta_grid, prior(theta_grid, 8, 2), label="Prior", lw=2, color='tab:blue', ls='--')
c_post, = ax.plot(theta_grid, post(theta_grid, 8, 8, 2), label=f"Posterior", lw=2, color='tab:blue')
ax.set_xlim(0, 1)
ax.set_ylim(0, None)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"Probability Density")
loc = plticker.MultipleLocator(base=0.1)
ax.xaxis.set_major_locator(loc)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


ax_a = fig.add_axes([0.15, 0.10, 0.65, 0.04])  # left, bottom, width, height
ax_b = fig.add_axes([0.15, 0.06, 0.65, 0.04])
ax_y = fig.add_axes([0.15, 0.02, 0.65, 0.04])

# create the sliders
sa = Slider(
    ax_a, "a", 0, 10,
    valinit=8, valstep=1,
)
sb = Slider(
    ax_b, "b", 0, 10,
    valinit=2, valstep=1,
)
sy = Slider(
    ax_y, "y", 0, n,
    valinit=8, valstep=1,
)

def update(val):
    a = sa.val
    b = sb.val
    y = sy.val
    c_prior.set_ydata(prior(theta_grid, a, b))
    c_post.set_ydata(post(theta_grid, y, a, b))
    fig.canvas.draw_idle()

sa.on_changed(update)
sb.on_changed(update)
sy.on_changed(update)

plt.show()
