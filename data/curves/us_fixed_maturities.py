"""
Generates the US zero yield curve, interpolated at fixed monthly maturies
"""
from data.utils import output_path
from data import us_zero_curve
import pandas as pd
import numpy as np


# Read the Data
data = us_zero_curve()

# ===== Interpolate Flat-Forward =====
curve = pd.DataFrame(data=data, columns=range(1, 121), index=data.index)

# generate the log-discount
curve = np.log(1 / (1 + curve)**(curve.columns / 12))

# linear interpolation
curve = curve.interpolate(method='index', axis=1, limit_area='inside')

# Back to rate
curve = (1 / np.exp(curve)) ** (12 / curve.columns) - 1

curve.columns = [f"{mat}m" for mat in curve.columns]

# ===== Save rates =====
curve.to_csv(output_path.joinpath('us monthly maturities.csv'))
