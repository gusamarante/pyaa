import numpy as np
import matplotlib.pyplot as plt

cases = {
    r"Case 1  $d=\alpha\kappa$": {
        "beta": 0.8,
        "pi": 0.1,
        "alpha": 0.8,
        "kappa": 0.3,
    },
    r"Case 2  $d=\kappa$": {
        "beta": 0.8,
        "pi": 0.02,
        "alpha": 0.8,
        "kappa": 0.3,
    },
}

start_beta = 0.9  # impatience factor
start_pi = 0.1  # Prob of state a
start_alpha = 0.7  # Marginal utility shock in state a
start_kappa = 0.3  # Moral cost of defaulting

n_steps = 200
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

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title(r"Lifetime Utility $U(d)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel(r"Debt $d$")

for name, params in cases.items():
    ax.plot(
        d_range,
        [
            utility(d, params['alpha'], params['kappa'], params['beta'], params['pi'])
            for d in d_range
        ],
        label=name,
    )

ax.legend(frameon=True, loc= "best")

plt.tight_layout()
plt.savefig(f'/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Problem Set 03/figures/item2 objevtive.pdf')
plt.show()
