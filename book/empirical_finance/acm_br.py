from utils import EMPIRICAL_FINANCE, color_palette
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from models import NominalACM
from data import curve_di
import matplotlib as mpl
import numpy as np
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=list(color_palette.values()))

size = 5

di = curve_di()
di = di.loc[:, :'121m'].dropna()

acm = NominalACM(
    curve=di,
)

# =====================================================
# ===== Chart - Observed, Fitted and Risk-Neutral =====
# =====================================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))


mat = "60m"
ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(di[mat], label='Observed', color=color_palette['blue'])
ax.plot(acm.miy[mat], label='Fitted', color=color_palette['yellow'])
ax.plot(acm.rny[mat], label='Risk-Neutral', color=color_palette['red'])
ax.set_title(f"5y DI Futures")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")


mat = "120m"
ax = plt.subplot2grid((1, 2), (0, 1), sharey=ax)
ax.plot(di[mat], label='Observed', color=color_palette['blue'])
ax.plot(acm.miy[mat], label='Fitted', color=color_palette['yellow'])
ax.plot(acm.rny[mat], label='Risk-Neutral', color=color_palette['red'])
ax.set_title(f"10y DI Futures")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(EMPIRICAL_FINANCE.joinpath("ACM BR - Observed, Fitted and Risk-Neutral.pdf"))
plt.show()
plt.close()


# ================================
# ===== Chart - Term Premium =====
# ================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(acm.tp["12m"], label='1y', color=color_palette['blue'])
ax.plot(acm.tp["24m"], label='2y', color=color_palette['red'])
ax.plot(acm.tp["60m"], label='5y', color=color_palette['green'])
ax.plot(acm.tp["120m"], label='10y', color=color_palette['yellow'])
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f"DI Futures Term Premium")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(EMPIRICAL_FINANCE.joinpath("ACM BR - Term Premium.pdf"))
plt.show()
plt.close()


# ============================================
# ===== Chart - Expected Return Loadings =====
# ============================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

curve_loadings = acm.pc_loadings_m
curve_loadings.index = [int(m) for m in curve_loadings.index.str[:-1]]
curve_loadings = curve_loadings * acm.pc_factors_m.std()

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(curve_loadings, label=curve_loadings.columns)
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f"Curve Loadings")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Maturity in Months")
ax.legend(frameon=True, loc="best")

er_loadings = acm.er_loadings
er_loadings.index = [int(m) for m in acm.er_loadings.index.str[:-1]]

ax = plt.subplot2grid((1, 2), (0, 1))
ax.plot(er_loadings, label=er_loadings.columns)
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f"Expected Return Loadings")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Maturity in Months")
ax.legend(frameon=True, loc="best")

plt.tight_layout()

plt.savefig(EMPIRICAL_FINANCE.joinpath("ACM BR - PC Loadings and Expected Return Loadings.pdf"))
plt.show()
plt.close()


# ===================================================
# ===== Chart - Term Premium vs Expected Return =====
# ===================================================
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(acm.tp['60m'], label='Term Premium')
ax.plot(acm.er_hist_d['60m'], label='Expected Return')
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f"5y DI Futures")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

ax = plt.subplot2grid((1, 2), (0, 1))
ax.plot(acm.tp['120m'], label='Term Premium')
ax.plot(acm.er_hist_d['120m'], label='Expected Return')
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f"10y DI Futures")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
ax.legend(frameon=True, loc="best")

plt.tight_layout()
plt.savefig(EMPIRICAL_FINANCE.joinpath("ACM BR - Term Premium VS Expected Return.pdf"))
plt.show()
plt.close()


# =============================================================
# ===== Chart - Cross Section of Expected Returns and Vol =====
# =============================================================
ret = acm.er_hist_d.iloc[-1]
ret.index = [int(m) for m in ret.index.str[:-1]]

std = acm.rx_m.std()
std.index = [int(m) for m in std.index.str[:-1]]

sharpe = ret / std

fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.plot(std.values * np.sqrt(12), 12 * ret.values, color=color_palette['blue'])
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel("Monthly Vol")
ax.set_ylabel("Monthly Return")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.tick_params(rotation=90, axis="x")

ax = plt.subplot2grid((1, 2), (0, 1))
ax.bar(sharpe.index, sharpe.values, color=color_palette['blue'], width=1)
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel("Maturity in Months")
ax.set_ylabel("Expected Sharpe Ratio")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)


plt.tight_layout()
plt.savefig(EMPIRICAL_FINANCE.joinpath("ACM BR - Cross Section Expected Returns.pdf"))
plt.show()
plt.close()


# ===================================================
# ===== Chart - Expected Return and Performance =====
# ===================================================
# TODO implement this
