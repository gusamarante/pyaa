import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
from utils import STOCHASTIC_CALCULUS


def get_srw_dist(n, t):
    nt = int(n * t)
    df = pd.DataFrame(
        data={
            'probs': binom.pmf(k=range(0, nt+1), n=nt, p=0.5),
            'srw': [(2*h - nt)/np.sqrt(n) for h in range(0, nt+1)],
        }
    )
    return df


# Chart
fig = plt.figure(figsize=(7 , 7))
t_plot = 0.2
possible_ns = [500, 250, 100, 20, 5]
for count, n_plot in enumerate(possible_ns):

    dist = get_srw_dist(n_plot, t_plot)

    if count == 0:
        ax = plt.subplot2grid((len(possible_ns), 1), (len(possible_ns) - count - 1, 0))
        ax.set_xlabel("Scaled Symetric Random Walk")
    else:
        ax = plt.subplot2grid((len(possible_ns), 1), (len(possible_ns) - count - 1, 0), sharex=ax)
        plt.tick_params('x', labelbottom=False)

    ax.bar(dist['srw'], dist['probs'],
           width=dist['srw'].diff(1).mean(),
           edgecolor='black',
           label=fr"$W_{{0.2}}^{{\left({n_plot}\right)}}$")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(STOCHASTIC_CALCULUS.joinpath("Brownian Motion - Limiting Distribution Convergence.pdf"))
plt.show()
plt.close()
