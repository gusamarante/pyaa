"""
Generates the DI curve, interpolated at fixed monthly maturies
"""
from data.utils import output_path
from data import raw_di
import pandas as pd
import numpy as np


# Read the Data
data = raw_di()

# ===== Interpolate Flat-Forward =====
curve = data.pivot(index='reference_date', columns='du', values='rate')

# generate the log-discount
curve = np.log(1 / (1 + curve)**(curve.columns / 252))

# linear interpolation
curve = curve.interpolate(method='index', axis=1, limit_area='inside')

# Back to rate
curve = (1 / np.exp(curve)) ** (252 / curve.columns) - 1

# Keep desired columns and drop NAs
max_months = int(curve.columns[-1] / 21)
desired_maturities = [21 * t for t in range(1, max_months)]
curve = curve[desired_maturities]
curve.columns = [f"{int(mat/21)}m" for mat in curve.columns]

# ===== Save rates =====
curve.to_csv(output_path.joinpath('di monthly maturities.csv'))
