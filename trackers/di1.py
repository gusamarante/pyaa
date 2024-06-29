"""
Builds Excess Return Indexes for the Brazilian DI Futures
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd

# User defined parameters
desired_duration = [i for i in range(1, 11)]  # in years
# desired_duration = [10]  # in years
rebalance_window = 2  # in months
last_year = 2024
start_date = '2008-01-01'  # when the 10y gets enough liquidity

# Read the Data
di = pd.DataFrame()

for year in tqdm(range(2006, last_year + 1), 'Reading files'):
    aux = pd.read_csv(f'input data/dados_di1 {year}.csv', sep=';')
    di = pd.concat([di, aux])

di = di.drop('Unnamed: 0', axis=1)
di['reference_date'] = pd.to_datetime(di['reference_date'])
di['maturity_date'] = pd.to_datetime(di['maturity_date'])

# Set up
dates2loop = pd.to_datetime(di['reference_date'].unique()).sort_values()
dates2loop = dates2loop[dates2loop >= start_date]

df_tracker = pd.DataFrame()  # To save all the final trackers

# ===== Fixed Duration =====
for dd in desired_duration:
    df_bt = pd.DataFrame()

    # First date
    aux_data = di[di['reference_date'] == dates2loop[0]]
    aux_data = aux_data[aux_data['contract'].str.contains("F")]
    aux_data = aux_data[aux_data['du'] >= rebalance_window * 21]
    aux_data = aux_data.sort_values('du')
    aux_data = aux_data.set_index('contract')

    dur_idx = aux_data['du'].searchsorted(dd * 252)
    a = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].values
    x = (dd * 252 - a[1]) / (a[0] - a[1])  # Ammount of contract 1
    current_cont1, current_cont2 = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].index
    df_bt.loc[dates2loop[0], 'contract 1'] = current_cont1
    df_bt.loc[dates2loop[0], 'contract 2'] = current_cont2
    df_bt.loc[dates2loop[0], 'du 1'] = aux_data.loc[current_cont1, 'du']
    df_bt.loc[dates2loop[0], 'du 2'] = aux_data.loc[current_cont2, 'du']
    notional_start = x * aux_data.loc[current_cont1, 'theoretical_price'] + (1 - x) * aux_data.loc[current_cont2, 'theoretical_price']
    df_bt.loc[dates2loop[0], 'quantity 1'] = x * notional_start / aux_data.loc[current_cont1, 'theoretical_price']
    df_bt.loc[dates2loop[0], 'quantity 2'] = (1 - x) * notional_start / aux_data.loc[current_cont2, 'theoretical_price']
    price1 = aux_data.loc[current_cont1, 'theoretical_price']
    price2 = aux_data.loc[current_cont2, 'theoretical_price']
    df_bt.loc[dates2loop[0], 'Notional'] = df_bt.loc[dates2loop[0], 'quantity 1'] * price1 + df_bt.loc[dates2loop[0], 'quantity 2'] * price2

    next_rebalance_date = dates2loop[0] + pd.DateOffset(months=rebalance_window)

    # Loop for other dates
    paired_dates = zip(dates2loop[1:], dates2loop[:-1])
    for date, datem1 in tqdm(paired_dates, f'ERI DI1 {dd}y'):
        # get available contracts
        aux_data = di[di['reference_date'] == date]
        aux_data = aux_data[aux_data['contract'].str.contains("F")]
        aux_data = aux_data.set_index('contract')
        aux_data = aux_data.sort_values('du')

        pnl = df_bt.loc[datem1, 'quantity 1'] * aux_data.loc[df_bt.loc[datem1, 'contract 1'], 'pnl'] \
            + df_bt.loc[datem1, 'quantity 2'] * aux_data.loc[df_bt.loc[datem1, 'contract 2'], 'pnl']

        df_bt.loc[date, 'Notional'] = df_bt.loc[datem1, 'Notional'] + pnl

        # TODO add transaction cost here

        if date >= next_rebalance_date:
            # rebalance to target duration
            aux_data = aux_data[aux_data['du'] >= rebalance_window * 21]
            dur_idx = aux_data['du'].searchsorted(dd * 252)
            a = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].values
            x = (dd * 252 - a[1]) / (a[0] - a[1])  # Ammount of contract 1
            current_cont1, current_cont2 = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].index

            df_bt.loc[date, 'contract 1'] = current_cont1
            df_bt.loc[date, 'contract 2'] = current_cont2
            df_bt.loc[date, 'quantity 1'] = x * df_bt.loc[date, 'Notional'] / aux_data.loc[current_cont1, 'theoretical_price']
            df_bt.loc[date, 'quantity 2'] = (1 - x) * df_bt.loc[date, 'Notional'] / aux_data.loc[current_cont2, 'theoretical_price']
            df_bt.loc[date, 'du 1'] = aux_data.loc[current_cont1, 'du']
            df_bt.loc[date, 'du 1'] = aux_data.loc[current_cont2, 'du']

            # set next rebalance date
            next_rebalance_date = date + pd.DateOffset(months=rebalance_window)

        else:
            df_bt.loc[date, 'contract 1'] = df_bt.loc[datem1, 'contract 1']
            df_bt.loc[date, 'contract 2'] = df_bt.loc[datem1, 'contract 2']
            df_bt.loc[date, 'quantity 1'] = df_bt.loc[datem1, 'quantity 1']
            df_bt.loc[date, 'quantity 2'] = df_bt.loc[datem1, 'quantity 2']
            df_bt.loc[date, 'du 1'] = df_bt.loc[datem1, 'du 1']
            df_bt.loc[date, 'du 2'] = df_bt.loc[datem1, 'du 2']

    df_tracker = pd.concat([df_tracker, df_bt['Notional'].rename(f"DI {dd}y")], axis=1)

# Standardize the tracker
df_tracker = 100 * df_tracker / df_tracker.iloc[0]

# ===== Save Trackers =====
filename = r'output data/trackers_di1.xlsx'
with pd.ExcelWriter(filename) as writer:
    df_tracker.to_excel(writer, sheet_name="Trackers DI1")
