"""
Chart of the CRRA utility as function of consumption, for different values of gamma
U(c) = (c**(1-gamma) - 1) / (1-gamma)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getuser

# Generate functions
c = np.linspace(0, 3, num=121)

gamma = 0
gamma0 = (c**(1 - gamma) - 1) / (1 - gamma)

gamma = 0.5
gamma05 = (c**(1 - gamma) - 1) / (1 - gamma)

# gamma = 1
gamma1 = np.log(c)

gamma = 2
gamma2 = (c**(1 - gamma) - 1) / (1 - gamma)

# Figure
plt.figure()

plt.plot(c, gamma0, linewidth=1.5, linestyle="-", label=r'$\gamma=0$')
plt.plot(c, gamma05, linewidth=1.5, linestyle="-", label=r'$\gamma=0.5$')
plt.plot(c, gamma1, linewidth=1.5, linestyle="-", label=r'$\gamma=1$')
plt.plot(c, gamma2, linewidth=1.5, linestyle="-", label=r'$\gamma=2$')

plt.axhline(0, linewidth=0.75, color='black')

ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

ax.set(title='CRRA Utility',
       xlabel='Consumption',
       ylabel='Utility')

plt.ylim(-2, 2.5)
plt.xlim(0, 3)

plt.legend(loc='lower right')
plt.tight_layout()
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Insper - Asset Allocation/Figures")
plt.savefig(file_path.joinpath("CRRA Utility.pdf"))
plt.show()
plt.close()
