"""
Run the backtest based on pre-computed PCs
"""
import matplotlib.ticker as plticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from utils import Performance
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import sqlite3
import getpass

# User Defined Parameters
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2024')

sd_enter = 1.5
sd_exit = 0.5

# Create Connection to DB
db_file = save_path.joinpath(r"di1pca.db")
with sqlite3.connect(db_file) as conn:
    # Query all the raw data
    query = """
    SELECT * FROM di1_raw;
    """
    df_raw = pd.read_sql(query, conn)

    # Query the PCs
    query = """
        SELECT reference_date, pc, pc_value FROM di1_pca
        where window_type = 'rolling 5y';
        """
    df_pc = pd.read_sql(query, conn)

    # Query the PCs' variance
    query = """
            SELECT reference_date, pc, pc_variance FROM di1_variance
            where window_type = 'rolling 5y';
            """
    df_sd = pd.read_sql(query, conn)

# Organize the raw data
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])

# Organize the PCs
df_pc['reference_date'] = pd.to_datetime(df_pc['reference_date'])
df_pc = df_pc.pivot(index='reference_date', columns='pc', values='pc_value')

# Organize the SD of the PCs
df_sd['reference_date'] = pd.to_datetime(df_sd['reference_date'])
df_sd = df_sd.pivot(index='reference_date', columns='pc', values='pc_variance')
df_sd = np.sqrt(df_sd)

# Build the interpolated curve
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# Build DV01
df_dv01 = df_raw.pivot(index='reference_date', columns='du', values='dv01')
df_dv01 = df_dv01 * 10_000  # PCA-DV01 requires move per unit of PC, so the DV01 has to be per unit of rate


# ====================================
# ===== Generate Signal Decision =====
# ====================================
dates2loop = df_pc.index
backtest = pd.DataFrame()  # To save everything from the backtest

position = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3'])
position.loc[dates2loop[0]] = [0, 0, 0]

dates2loop = zip(dates2loop[1:], dates2loop[:-1])
for d, dm1 in tqdm(dates2loop, "Generating Decisions"):

    # Where the signal stands today
    cond_above = df_pc.loc[d] > sd_enter * df_sd.loc[d]
    cond_below = df_pc.loc[d] < - sd_enter * df_sd.loc[d]
    cond_middle = (~cond_above) & (~cond_below)

    cond_exit_short = df_pc.loc[d] < sd_exit * df_sd.loc[d]
    cond_exit_long = df_pc.loc[d] > - sd_exit * df_sd.loc[d]

    cond_long = position.loc[dm1] == 1
    cond_short = position.loc[dm1] == -1
    cond_neutral = position.loc[dm1] == 0

    # --- Decision ---
    # PC above threshold - Sell it
    cond1 = cond_above
    # PC below threshold - Buy it
    cond2 = cond_below
    # Already short, not below exit - Stay short
    cond3 = cond_short & (~cond_exit_short)
    # Already long, not above exit - Stay long
    cond4 = cond_long & (~cond_exit_long)
    # Was short, now exit to neutral
    cond5 = cond_short & cond_exit_short
    # Was long, now exit to neutral
    cond6 = cond_long & cond_exit_long
    # Anything else, stay neutral
    cond7 = ~(cond1 | cond2 | cond3 | cond4 | cond5 | cond6)

    position.loc[d] = (cond1 | cond3) * (-1) + (cond2 | cond4) * 1 + (cond5 | cond6 | cond7) * 0


# ==================================
# ===== Chart of the Decisions =====
# ==================================
fig = plt.figure(figsize=(7 * (16 / 9), 7))
fig.suptitle(
    "DI1 PCA - Decision Triggers",
    fontsize=16,
    fontweight="bold",
)

# Upper left
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
pc = 'PC 2'
ax.fill_between(
    x=df_sd.index,
    y1=-sd_enter * df_sd[pc].values,
    y2=sd_enter * df_sd[pc].values,
    label=f"Entry Level {sd_enter}$\sigma$",
    alpha=0.2,
    color="green",
    linewidth=0,
)
ax.fill_between(
    x=df_sd.index,
    y1=-sd_exit * df_sd[pc].values,
    y2=sd_exit * df_sd[pc].values,
    label=f"Exit Level {sd_exit}$\sigma$",
    alpha=0.3,
    color="green",
    linewidth=0,
)
ax.plot(df_pc[pc], label=pc, color='darkgreen')
ax.set_title(f"{pc}")
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
loc = plticker.MultipleLocator(base=0.1)
ax.yaxis.set_major_locator(loc)
ax.legend(frameon=True, loc="upper left")

# Upper right
ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
pc = 'PC 3'
ax.fill_between(
    x=df_sd.index,
    y1=-sd_enter * df_sd[pc].values,
    y2=sd_enter * df_sd[pc].values,
    label=f"Entry Level {sd_enter}$\sigma$",
    alpha=0.2,
    color="green",
    linewidth=0,
)
ax.fill_between(
    x=df_sd.index,
    y1=-sd_exit * df_sd[pc].values,
    y2=sd_exit * df_sd[pc].values,
    label=f"Exit Level {sd_exit}$\sigma$",
    alpha=0.3,
    color="green",
    linewidth=0,
)
ax.plot(df_pc[pc], label=pc, color='darkgreen')
ax.set_title(f"{pc}")
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
loc = plticker.MultipleLocator(base=0.1)
ax.yaxis.set_major_locator(loc)
ax.legend(frameon=True, loc="upper left")

# lower left
ax = plt.subplot2grid((3, 2), (2, 0))
pc = 'PC 2'
ax.plot(position[pc], label=pc, color='darkgreen')
ax.set_title(f"{pc} - Decision")
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
loc = plticker.MultipleLocator(base=1)
ax.yaxis.set_major_locator(loc)

# lower right
ax = plt.subplot2grid((3, 2), (2, 1))
pc = 'PC 3'
ax.plot(position[pc], label=pc, color='darkgreen')
ax.set_title(f"{pc} - Decision")
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(rotation=90, axis="x")
loc = plticker.MultipleLocator(base=1)
ax.yaxis.set_major_locator(loc)

plt.tight_layout()
# plt.savefig(save_path.joinpath("????.pdf"))
plt.show()
plt.close()


# =======================================
# ===== Portfolio Building Function =====
# =======================================
def get_portfolio(current_date, pcadv01):
    """
    given a date and the desired exposition vector, returns the chosen contracts
    """

    # Query the factor loadings
    with sqlite3.connect(db_file) as dbcon:
        sql = f"""
        SELECT du, pc, loading FROM di1_loadings 
        WHERE window_type = 'rolling 5y'
        AND reference_date = '{current_date}';
        """
        df_loadings = pd.read_sql(sql, dbcon)

    df_loadings = df_loadings.pivot(index='du', columns='pc', values='loading')

    # TODO add liquidity filter here... eventually

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[3*21 <= available_maturities]
    available_maturities = available_maturities[5*252 >= available_maturities]

    aux_pcadv01 = df_loadings.loc[available_maturities]
    aux_pcadv01 = aux_pcadv01.multiply(df_dv01.loc[current_date, available_maturities], axis=0)

    # Choose 3 contracts
    vertices_du = aux_pcadv01.idxmax().sort_values().values

    selected_portfolio = pd.DataFrame(index=vertices_du)
    cond_date = df_raw['reference_date'] == current_date
    cond_du = df_raw['du'].isin(vertices_du)
    current_data = df_raw[cond_date & cond_du].sort_values('du')

    selected_portfolio['contracts'] = current_data['contract'].values
    selected_portfolio['pu'] = current_data['theoretical_price'].values
    selected_portfolio[['Loadings 1', 'Loadings 2', 'Loadings 3']] = df_loadings.loc[vertices_du]
    selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']] = aux_pcadv01.loc[vertices_du]

    coeff = selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']].T.values
    constants = np.array(pcadv01)
    selected_portfolio['quantities'] = np.linalg.inv(coeff) @ constants

    return selected_portfolio


# ====================
# ===== Backtest =====
# ====================
dates2loop = df_pc.index
backtest = pd.DataFrame()  # To save everything from the backtest
dates2loop = zip(dates2loop[1:], dates2loop[:-1])
has_notional = False
notional  = 0


for d, dm1 in tqdm(dates2loop, "Backtesting"):

    desired_dv01 = position.loc[dm1].values * np.array([0, 1000, 5000])
    port = get_portfolio(dm1, desired_dv01)  # LAG FOR INFORMATION
    port = port.set_index('contracts')

    if (port['quantities'] == 0).all() and not has_notional:
        continue
    elif (port['quantities'] == 0).all() and has_notional:
        pass
    else:
        notional = (port['quantities'] * port['pu']).sum()
        has_notional = True

    cond_date = df_raw['reference_date'] == d
    cond_contracts = df_raw['contract'].isin(port.index)
    mtm = df_raw[cond_date & cond_contracts][['contract', 'pnl']].set_index('contract')['pnl']
    pnl = (port['quantities'] * mtm).sum()
    backtest.loc[d, 'pnl'] = pnl


fd = backtest.index[0]
backtest.loc[fd, 'pnl'] = backtest.loc[fd, 'pnl'] + notional
backtest['pnl'] = backtest['pnl'].cumsum()


# ============================
# ===== Evaluate Results =====
# ============================
perf = Performance(backtest)
print(perf.table)


backtest['pnl'].plot()
plt.show()


# TODO Control rebalance dates
