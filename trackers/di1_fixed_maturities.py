import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time

tic = time()

# User defined parameters
desired_maturities = [21 * t for t in range(1, 15 * 12 + 1)] # in DUs
last_year = 2023
start_date = '2006-01-01'

# Read the Data
data = pd.DataFrame()

for year in tqdm(range(2006, last_year + 1), 'Reading files'):
    aux = pd.read_csv(f'input data/dados_di1 {year}.csv', sep=';')
    data = pd.concat([data, aux])

data['reference_date'] = pd.to_datetime(data['reference_date'])
data['maturity_date'] = pd.to_datetime(data['maturity_date'])
data = data.drop('Unnamed: 0', axis=1)


# ===== Interpolate Flat-Forward =====
curve = data.pivot(index='reference_date', columns='du', values='rate')
curve = curve.drop(0, axis=1)

# generate the log-discount
curve = np.log(1 / (1 + curve)**(curve.columns / 252))

# linear interpolation
curve = curve.interpolate(method='index', axis=1, limit_area='inside')

# Back to rate
curve = (1 / np.exp(curve)) ** (252 / curve.columns) - 1

# Kep desired columns and drop NAs
curve = curve[desired_maturities]
curve = curve.dropna(how='any', axis=1)
curve = curve.dropna()


# ===== Save rates =====
filename = r'output data/di fixed maturities.xlsx'
with pd.ExcelWriter(filename) as writer:
    curve.to_excel(writer, sheet_name="Rate")

minutes = round((time() - tic) / 60, 1)
print(f'DI Interpolation took {minutes} minutes')

