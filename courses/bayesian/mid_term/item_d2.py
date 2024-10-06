"""
This routine analyses the samples from the posterior, it does not generate them
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/samples.csv',
                 index_col=0)

# ===== CONVERGENCE =====
size = 5
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


# ===== POSTERIORS =====
df = df.iloc[1000:]

print(df[['alpha', 'beta', 'sig y', 'sig alpha', 'sig beta']].describe())

size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))

var_name = "alpha"
var_value = -4.53
ax = plt.subplot2grid((2, 3), (0, 0))
ax.hist(df[var_name], density=True, bins=40)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axvline(var_value, color='tab:orange', lw=2)
ax.set_title(r"$\alpha$")

var_name = "beta"
var_value = 0.48
ax = plt.subplot2grid((2, 3), (0, 1))
ax.hist(df[var_name], density=True, bins=40)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axvline(var_value, color='tab:orange', lw=2)
ax.set_title(r"$\beta$")

var_name = "sig y"
# var_value = 0.96
ax = plt.subplot2grid((2, 3), (0, 2))
ax.hist(df[var_name], density=True, bins=40)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.axvline(var_value, color='tab:orange', lw=2)  # TODO is this supposed to happen?
ax.set_title(r"$\sigma_y$")

var_name = "sig alpha"
var_value = 0.48
ax = plt.subplot2grid((2, 3), (1, 0))
ax.hist(df[var_name], density=True, bins=40)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axvline(var_value, color='tab:orange', lw=2)
ax.set_title(r"$\sigma_\alpha$")

var_name = "sig beta"
var_value = 0.0189
ax = plt.subplot2grid((2, 3), (1, 1))
ax.hist(df[var_name], density=True, bins=40)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.axvline(var_value, color='tab:orange', lw=2)
ax.set_title(r"$\sigma_\alpha$")


plt.tight_layout()
plt.savefig("/Users/gustavoamarante/Dropbox/Aulas/Doutorado - Bayesiana/Mid Term/figures/Item D posteriors.pdf")
plt.show()
plt.close()
