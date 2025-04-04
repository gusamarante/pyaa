import pandas as pd
import matplotlib.pyplot as plt


file_path = '/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Problem Set 02/Contas.xlsx'
size = 10

ls = {
    1: 'dotted',
    2: 'solid',
    3: 'dashed',  # or 'dashdot'
}

case = dict()
for c in range(1, 4):

    aux_df = pd.read_excel(
        io=file_path,
        sheet_name=f"Case {c}",
        index_col=0,
    )
    case[c] = aux_df



fig = plt.figure(figsize=(size * (1.2 / 1.41), size))

# Tax Rate
ax = plt.subplot2grid((4, 2), (0, 0))
ax.set_title(r"Tax Rate $\tau_t$")
for c in range(1, 4):
    ax.plot(case[c]['tau'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")  # Only one chart should hold the legend

# Tradeable Consumption
ax = plt.subplot2grid((4, 2), (0, 1))
ax.set_title(r"Tradeable Consumption $c_{t}^{M}$")
for c in range(1, 4):
    ax.plot(case[c]['cm'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
# ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# Non-Tradeable Consumption / Labor
ax = plt.subplot2grid((4, 2), (1, 0))
ax.set_title(r"Non-Tradeable Consumption / Labor $c_{t}^{N}=h_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['cn'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
# ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# price
ax = plt.subplot2grid((4, 2), (1, 1))
ax.set_title(r"Price $p_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['p'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
# ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# Debt
ax = plt.subplot2grid((4, 2), (2, 0))
ax.set_title(r"Debt $d_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['d'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# Trade Balance
ax = plt.subplot2grid((4, 2), (2, 1))
ax.set_title(r"Trade Balance $tb_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['tb'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# Current Account
ax = plt.subplot2grid((4, 2), (3, 0))
ax.set_title(r"Current Account $tb_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['ca'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

# Consumption Bundle
ax = plt.subplot2grid((4, 2), (3, 1))
ax.set_title(r"Consumption Bundle $c_{t}$")
for c in range(1, 4):
    ax.plot(case[c]['cb'], label=f'Case {c}', lw=2, ls=ls[c])

ax.axvline(0, color='black', lw=0.5)
# ax.axhline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)


plt.tight_layout()
plt.savefig(f'/Users/gamarante/Dropbox/Aulas/Doutorado - International Finance/Problem Set 02/figures/Q02 trajectories.pdf')
plt.show()
plt.close()
