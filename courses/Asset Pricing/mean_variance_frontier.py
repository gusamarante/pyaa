"""
Chart of mean-variance frontier restriction of assets that are priced by the stochastic discount factor
|E(R_{t+1}^{i})-R_{f}| <= (sigma_{m_{t+1}}) / (E_{t}[m_{t+1}]) sigma_{R_{i}}
Equation 1.17 from Cochrane's book
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
expected_c_growth = 0.02
sigma_c_growth = 0.02
gamma = 1.2
beta = 0.95
rf = -np.log(beta) + gamma * expected_c_growth - 0.5 * (gamma**2) * (sigma_c_growth ** 2)
em = 1 / (1 + rf)
sigma_m = np.sqrt((beta**2) * (gamma**2) * (sigma_c_growth**2))

# Generate functions
sigma_r = np.linspace(0, 0.5, num=20)

upper_bound = rf + (sigma_m / em) * sigma_r
lower_bound = rf - (sigma_m / em) * sigma_r

sigma_line = [0.2, 0.4]
ret_line = [rf + (sigma_m / em) * 0.2, rf + (sigma_m / em) * 0.2]

# Figure
plt.figure()
plt.plot(sigma_r, upper_bound, linewidth=1.5, linestyle="-", color='blue', label=r'Mean-Variance Frontier')
plt.plot(sigma_r, lower_bound, linewidth=1.5, linestyle="-", color='blue')
plt.plot(sigma_line, ret_line, linewidth=1.5, linestyle="--", color='red', marker='o')
plt.fill_between(sigma_r, lower_bound, upper_bound, color='blue',  alpha=.1)

ax = plt.gca()
ax.set(title='Mean-Variance Frontier',
       xlabel='$\sigma_{R^{i}}$',
       ylabel='$E(R^{i})$')

plt.xlim(0, 0.5)

# Annotation Slope
plt.annotate(r'Slope=$\frac{\sigma_{m}}{E(m)}$',  # text
             xy=(0.15, rf + (sigma_m / em) * 0.15),  # coordinate of the annotated point
             xycoords='data',  # unit of the coordinate of the annotated point
             xytext=(-90, 50),  # coordinate of the annotation text
             textcoords='offset points',  # unit of the coordinate of the annotation text (offset from point)
             fontsize=12,
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3,rad=.2"))

# Annotation Idiosyncratic risk
plt.annotate(r'Idiosyncratic Risk',  # text
             xy=(0.3, ret_line[0] - 0.001),  # coordinate of the annotated point
             xycoords='data',  # unit of the coordinate of the annotated point
             xytext=(0, 0),  # coordinate of the annotation text
             textcoords='offset points',  # unit of the coordinate of the annotation text (offset from point)
             fontsize=12,
             color='red',
             ha='center',
             va='center')

plt.tight_layout()
# plt.savefig('/Users/gustavoamarante/Dropbox/Aulas/QuantFin/Asset Pricing/figures/mean_variance_frontier.pdf')
plt.show()