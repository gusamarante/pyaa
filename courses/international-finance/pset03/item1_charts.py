import numpy as np
import matplotlib.pyplot as plt

beta = 0.9  # impatience factor
pi = 0.1  # Prob of state a
alpha = 0.7  # Marginal utility shock in state a
kappa = 0.3  # Moral cost of defaulting

n_steps = 100
size = 5

def utility(d, q, alpha, kappa, beta, pi):
    """
    Lifetime utility
    """
    if d <= alpha * kappa:
        u = q * d + beta * (pi * ((1 - d) / alpha) + (1 - pi) * (1 - d))
    else:
        u = q * d + beta * (pi * (1 / alpha - kappa) + (1 - pi) * (1 - d))

    return u


def value_fun(q, alpha, kappa, beta, pi):
    """
    Value function of the consumer problem
    """
    if q <= beta:
        v = beta * (pi/alpha + 1 - pi)
    else:
        v = q * kappa + beta * (pi/alpha + 1 - pi - kappa)

    return v


d_range = np.arange(
    start=0,
    stop=kappa + kappa / n_steps,
    step=kappa / n_steps,
)
q_max = 1
q_inc = q_max / n_steps
q_range = np.arange(start=0.7, stop=q_max + q_inc, step=q_inc)

lim1 = beta*(1-pi)
lim2 = beta
lim3 = beta * (pi / alpha + 1 - pi)
q2plot = [
    lim1 - 0.015,
    lim1,
    (lim1 + lim2) / 2,
    lim2,
    (lim2 + lim3) / 2,
    lim3,
    lim3 + 0.015,
]

# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(size * (16 / 9), size))

# Lifetime Utility
ax_lu = plt.subplot2grid((1, 2), (0, 0))
ax_lu.set_title(r"Lifetime Utility $U(d)$")
for q in q2plot:
    ax_lu.plot(
        d_range,
        [
            utility(d, q, alpha, kappa, beta, pi)
            for d in d_range
        ],
        label=fr"$q={round(q, 2)}$",
    )
ax_lu.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_lu.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_lu.set_xlabel(r"Debt $d$")
ax_lu.legend(frameon=True, loc='best')

# Value Function
ax_vf = plt.subplot2grid((1, 2), (0, 1), sharey=ax_lu)
ax_vf.set_title(r"Value Function $V(q)$")
ax_vf.plot(
    q_range,
    [
        value_fun(q, alpha, kappa, beta, pi)
        for q in q_range
    ],
    color='dimgrey',
    ls='--',
)
for q in q2plot:
    ax_vf.plot([q], [value_fun(q, alpha, kappa, beta, pi)],
               marker="o", ls=None, markeredgecolor="black")

ax_vf.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_vf.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax_vf.set_xlabel(r"Price of Bond $q$")

plt.tight_layout()
plt.savefig(f'/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Problem Set 03/figures/item1 objevtive and value functions.pdf')
plt.show()
plt.close()
