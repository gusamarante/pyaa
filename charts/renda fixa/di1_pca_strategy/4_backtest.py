"""
Run the backtest based on pre-computed PCs
"""
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


# =======================================
# ===== Portfolio Building Function =====
# =======================================
def get_portfolio(current_date, pcadv01):
    """
    given a date and the desired exposition vector, returns the chosen contracts
    """

    # Query the factor loadings
    with sqlite3.connect(db_file) as dbcon:
        sql = """
                SELECT reference_date, du, pc, loading FROM di1_loadings
                where window_type = 'rolling 5y';
                """
        df_loadings = pd.read_sql(sql, dbcon)

    df_loadings['reference_date'] = pd.to_datetime(df_loadings['reference_date'])
    df_load1 = df_loadings[df_loadings['pc'] == 'PC 1']
    df_load1 = df_load1.pivot(index='reference_date', columns='du',
                              values='loading')

    # TODO query what is needed here
    current_date = pd.to_datetime(current_date)

    current_loadings = df_loadings[df_loadings['reference_date'] == current_date]

    # TODO add liquidity filter here... eventually

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[df_curve.columns.min() <= available_maturities]
    available_maturities = available_maturities[df_curve.columns.max() >= available_maturities]

    aux_pcadv01 = current_loadings.loc[available_maturities]
    aux_pcadv01 = aux_pcadv01.multiply(df_dv01.loc[current_date, available_maturities], axis=0)
    aux_pcadv01 = aux_pcadv01.astype(float)  # TODO is this needed?

    # Choose 3 contracts
    vertices_du = aux_pcadv01.idxmax().sort_values().values

    selected_portfolio = pd.DataFrame(index=vertices_du)
    cond_date = df_raw['reference_date'] == current_date
    cond_du = df_raw['du'].isin(vertices_du)
    current_data = df_raw[cond_date & cond_du].sort_values('du')

    selected_portfolio['contracts'] = current_data['contract'].values
    selected_portfolio['pu'] = current_data['theoretical_price'].values
    selected_portfolio[['Loadings 1', 'Loadings 2', 'Loadings 3']] = current_loadings.loc[vertices_du]
    selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']] = aux_pcadv01.loc[vertices_du]

    coeff = selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']].T.values
    constants = np.array(pcadv01)
    selected_portfolio['quantities'] = np.linalg.inv(coeff) @ constants

    return selected_portfolio


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

# TODO Chart of the signal, with ranges and decision

# TODO Backtest
#  when signal changes, rebuild and set rebalance for m months
