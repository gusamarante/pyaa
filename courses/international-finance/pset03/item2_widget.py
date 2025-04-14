import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import Slider

start_beta = 0.9  # impatience factor
start_pi = 0.1  # Prob of state a
start_alpha = 0.7  # Marginal utility shock in state a
start_kappa = 0.3  # Moral cost of defaulting
start_q = 0.6

n_steps = 100
size = 5

def utility(d, alpha, kappa, beta, pi):

    if d <= alpha*kappa:
        u = d + beta * (pi * ((1 - d) / alpha) + (1 - pi) * (1 - d))
    else:
        u = (1 - pi) * d + beta * (pi * (1 / alpha - kappa) + (1 - pi) * (1 - d))

    return u


d_range = np.arange(start=0, stop=start_kappa + start_kappa/n_steps, step=start_kappa/n_steps)


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (20 / 9), size))

# Lifetime Utility
ax_lu = plt.subplot2grid((5, 3), (0, 0), rowspan=5)
ax_lu.set_title(r"Lifetime Utility $U(d)$")
c_lu, = ax_lu.plot(
    d_range,
    [
        utility(d, start_alpha, start_kappa, start_beta, start_pi)
        for d in d_range
    ],
)
ax_lu.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_lu.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_lu.set_xlabel(r"Debt $d$")


# Sliders
ax_beta = plt.subplot2grid((5, 3), (0, 2))
sbeta = Slider(
    ax=ax_beta,
    label=r"$\beta$",
    valmin=0,
    valmax=1,
    valinit=start_beta,
    valstep=0.005,
)
ax_pi = plt.subplot2grid((5, 3), (1, 2))
spi = Slider(
    ax=ax_pi,
    label=r"$\pi$",
    valmin=0,
    valmax=1,
    valinit=start_pi,
    valstep=0.005,
)
ax_alpha = plt.subplot2grid((5, 3), (2, 2))
salpha = Slider(
    ax=ax_alpha,
    label=r"$\alpha$",
    valmin=0.01,
    valmax=1,
    valinit=start_alpha,
    valstep=0.005,
)
ax_kappa = plt.subplot2grid((5, 3), (3, 2))
skappa = Slider(
    ax=ax_kappa,
    label=r"$\kappa$",
    valmin=0,
    valmax=1,
    valinit=start_kappa,
    valstep=0.005,
)

def update(val):
    nbeta = sbeta.val
    npi = spi.val
    nalpha = salpha.val
    nkappa = skappa.val

    cond = nbeta * (npi / nalpha + 1 - npi) < 1 - npi
    print(cond)

    c_lu.set_ydata([
        utility(d, nalpha, nkappa, nbeta, npi)
        for d in d_range
    ])
    ax_lu.relim()
    ax_lu.autoscale_view()
    fig.canvas.draw_idle()


sbeta.on_changed(update)
spi.on_changed(update)
salpha.on_changed(update)
skappa.on_changed(update)

plt.tight_layout()
plt.show()