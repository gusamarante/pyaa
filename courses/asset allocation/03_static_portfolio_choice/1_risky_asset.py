"""
Combination of 1 risky asset + risk-free asset

Color Palette
3333B2 - Latex Blue
191959 - Darker Blue
0B6E4F - Green
FFBA08 - Yellow
F25F5C - Red
"""

import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser

rf = 0.02
sigma = 0.2
sharpe = 0.3
mu = rf + sharpe * sigma


# =================
# ===== CHART =====
# =================
fig = plt.figure(figsize=(5 * (10 / 7.3), 5))

# Industrial Metals
ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(0, rf, label='Risk-Free', color="#FFBA08", s=50)
ax.scatter(sigma, mu, label='Risky', color="#F25F5C", s=50)
ax.axline([0, rf], [sigma, mu], color="#3333B2", label="Possible Combinations", lw=2, zorder=-1)

ax.set_xlim((0, None))
ax.set_ylim((0, None))
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="best")


plt.tight_layout()

file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")
plt.savefig(file_path.joinpath("Static Portfolio Choice - One risky plus riskless.pdf"))
plt.show()
plt.close()
