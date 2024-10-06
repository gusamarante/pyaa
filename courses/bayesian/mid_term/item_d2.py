"""
This routine analyses the samples from the posterior, it does not generate them
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv',
                 index_col=0)
print(df)

# ===== CONVERGENCE =====
size = 5  # TODO set this
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((2, 2), (0, 0))
ax.plot(df['alpha'].iloc[:1200])
ax.axhline(0, color='black', lw=0.5)
ax.axvspan(0, 1000, color='grey', alpha=0.3)
ax.set_title(r"$\alpha$")
ax.set_xlabel("Number of iterations")

ax = plt.subplot2grid((2, 2), (1, 0))
ax.plot(df['sig alpha'].iloc[:1200])
ax.axhline(0, color='black', lw=0.5)
ax.axvspan(0, 1000, color='grey', alpha=0.3)
ax.set_title(r"$\sigma_\alpha$")
ax.set_xlabel("Number of iterations")

ax = plt.subplot2grid((2, 2), (0, 1))
ax.plot(df['beta'].iloc[:1200])
ax.axhline(0, color='black', lw=0.5)
ax.axvspan(0, 1000, color='grey', alpha=0.3)
ax.set_title(r"$\beta$")
ax.set_xlabel("Number of iterations")

ax = plt.subplot2grid((2, 2), (1, 1))
ax.plot(df['sig beta'].iloc[:1200])
ax.axhline(0, color='black', lw=0.5)
ax.axvspan(0, 1000, color='grey', alpha=0.3)
ax.set_title(r"$\sigma_\beta$")
ax.set_xlabel("Number of iterations")

plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item D convergence.pdf")
plt.show()
plt.close()

# TODO plot posteriors of selected parameters, comparing with OLS
# TODO tables of selected parameters


