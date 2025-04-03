"""
Builds Excess Return Indexes for the Brazilian DI Futures
"""
from data.utils import output_path
from data import raw_di
from tqdm import tqdm
import pandas as pd

# User defined parameters
desired_duration = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # in years
rebalance_window = 2  # in months
start_date = '2008-01-01'  # when the 10y gets enough liquidity
index_start = 100

# Read the Data
di = raw_di()

# Set up
dates2loop = pd.to_datetime(di['reference_date'].unique())
dates2loop = dates2loop[dates2loop >= start_date]

df_tracker = pd.DataFrame()  # To save all the final trackers

# ===== Backtests for each fixed duration =====
for dd in desired_duration:
    df_bt = pd.DataFrame()

    # First date
    aux_data = di[di['reference_date'] == dates2loop[0]]
    aux_data = aux_data[aux_data['contract'].str.contains("F")]
    aux_data = aux_data[aux_data['du'] >= rebalance_window * 21]
    aux_data = aux_data.sort_values('du')
    aux_data = aux_data.set_index('contract')

    dur_idx = aux_data['du'].searchsorted(dd * 252)
    if dur_idx == 0:  # shortest contract is shorter than desired duration
        a = aux_data['du'].iloc[0]
        x = (dd * 252) / a  # Ammount of contract 1
        y = 0  # ammount of contract 2
        current_cont1, current_cont2 = aux_data['du'].index[0], None

    else:
        a = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].values
        x = (dd * 252 - a[1]) / (a[0] - a[1])  # Ammount of contract 1
        y = 1 - x
        current_cont1, current_cont2 = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].index

    df_bt.loc[dates2loop[0], 'contract 1'] = current_cont1
    df_bt.loc[dates2loop[0], 'contract 2'] = current_cont2

    df_bt.loc[dates2loop[0], 'quantity 1'] = (index_start * x) / aux_data['theoretical_price'].get(current_cont1, default=1)
    df_bt.loc[dates2loop[0], 'quantity 2'] = (index_start * y) / aux_data['theoretical_price'].get(current_cont2, default=1)  # Only defaults to 1 when current_cont2 is None and y is 0
    df_bt.loc[dates2loop[0], 'Notional'] = index_start

    next_rebalance_date = dates2loop[0] + pd.DateOffset(months=rebalance_window)

    # Loop for other dates
    paired_dates = zip(dates2loop[1:], dates2loop[:-1])
    for date, datem1 in tqdm(paired_dates, f'ERI DI1 {dd}y'):
        # Compute PnL before rebalance
        aux_data = di[di['reference_date'] == date]
        aux_data = aux_data[aux_data['contract'].str.contains("F")]
        aux_data = aux_data.set_index('contract')
        aux_data = aux_data.sort_values('du')

        pnl = df_bt.loc[datem1, 'quantity 1'] * aux_data['pnl'].get(df_bt.loc[datem1, 'contract 1'], default=0) \
            + df_bt.loc[datem1, 'quantity 2'] * aux_data['pnl'].get(df_bt.loc[datem1, 'contract 2'], default=0)

        df_bt.loc[date, 'Notional'] = df_bt.loc[datem1, 'Notional'] + pnl

        # Rebalance or Hold
        if date >= next_rebalance_date:
            # rebalance to target duration
            aux_data = aux_data[aux_data['du'] >= rebalance_window * 21]
            dur_idx = aux_data['du'].searchsorted(dd * 252)

            if dur_idx == 0:  # shortest contract is shorter than desired duration
                a = aux_data['du'].iloc[0]
                x = (dd * 252) / a  # Ammount of contract 1
                y = 0  # ammount of contract 2
                current_cont1, current_cont2 = aux_data['du'].index[0], None

            else:
                a = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].values
                x = (dd * 252 - a[1]) / (a[0] - a[1])  # Ammount of contract 1
                y = 1 - x
                current_cont1, current_cont2 = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].index

            df_bt.loc[date, 'contract 1'] = current_cont1
            df_bt.loc[date, 'contract 2'] = current_cont2
            df_bt.loc[date, 'quantity 1'] = (x * df_bt.loc[date, 'Notional']) / aux_data['theoretical_price'].get(current_cont1, default=1)
            df_bt.loc[date, 'quantity 2'] = (y * df_bt.loc[date, 'Notional']) / aux_data['theoretical_price'].get(current_cont2, default=1)

            # set next rebalance date
            next_rebalance_date = date + pd.DateOffset(months=rebalance_window)

        else:
            df_bt.loc[date, 'contract 1'] = df_bt.loc[datem1, 'contract 1']
            df_bt.loc[date, 'contract 2'] = df_bt.loc[datem1, 'contract 2']
            df_bt.loc[date, 'quantity 1'] = df_bt.loc[datem1, 'quantity 1']
            df_bt.loc[date, 'quantity 2'] = df_bt.loc[datem1, 'quantity 2']

    df_tracker = pd.concat([df_tracker, df_bt['Notional'].rename(f"DI {dd}y")], axis=1)

# Standardize the tracker
df_tracker = 100 * df_tracker / df_tracker.iloc[0]

# ===== Save Trackers =====
df_tracker.to_csv(output_path.joinpath('trackers_di1.csv'))
