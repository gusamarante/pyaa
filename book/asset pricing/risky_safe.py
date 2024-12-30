"""
Grabs 3 brazilian assets and builds the min variance frontier.
"""
# TODO systematize the consumption of total returns of FIPs
from allocation import MeanVar
from utils import Performance
from data import SGS, trackers_ntnb
import pandas as pd

# =====================
# ===== Read Data =====
# =====================
# IVVB
filepath = '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Personal Portfolio/data/ETFs.xlsx'  # Mac
# filepath = r"C:\Users\gamarante\Dropbox\Personal Portfolio\data\ETFs.xlsx"  # Work
ivvb = pd.read_excel(filepath, index_col=0, na_values=['#N/A N/A'], sheet_name='values')
ivvb = ivvb['IVVB11 BZ Equity'].rename('IVVB')
ivvb = ivvb.dropna()

# BDIV
filepath = '/Users/gustavoamarante/Library/CloudStorage/Dropbox/Personal Portfolio/data/FIPs.xlsx'  # Mac
# filepath = r"C:\Users\gamarante\Dropbox\Personal Portfolio\data\FIPs.xlsx"  # Work
bdiv = pd.read_excel(filepath, index_col=0, sheet_name='Trackers')
bdiv = bdiv['BDIV']
bdiv = bdiv.dropna()

# NTNB
ntnb = trackers_ntnb()
ntnb = ntnb['NTNB 10y']

# Total return index
tri = pd.concat([ivvb, bdiv, ntnb], axis=1)

# ==================================
# ===== Compute Excess Returns =====
# ==================================
sgs = SGS()
cdi = sgs.fetch(series_id={12: 'CDI'})
cdi = cdi['CDI'] / 100

eri = tri.pct_change(1)
eri = eri.sub(cdi, axis=0)
eri = eri.dropna(how='all')
eri = (1 + eri).cumprod()

# ==========================
# ===== Excess Returns =====
# ==========================
# Performance
perf_tri = Performance(tri, skip_dd=True)
print(perf_tri.table)

perf_eri = Performance(eri, skip_dd=True)
print(perf_eri.table)

# Correlation
corr = eri.resample('M').last().pct_change(1).corr()
print(corr)

# Min Variance Parameters
mu = perf_tri.returns_ann
cov = eri.resample('M').last().pct_change(1).cov() * 12
rf = (1 + cdi.iloc[-1]) ** 252 - 1

mv = MeanVar(mu, cov, rf=rf, risk_aversion=100)

print('Max Sharpe', mv.risky_weights)
print('Min Var', mv.mv_weights)

mv.plot(mvfnoss=False)
