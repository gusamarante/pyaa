import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(io=r"C:\Users\gamarante\Downloads\Book1.xlsx",
                   index_col=0)

df = df['IVVB11 BZ Equity'].dropna()
ret = df.pct_change(1)

k_fast = 21 * 6
k_slow = 21 * 12
df_fast = ret.rolling(k_fast).mean().shift(1)  # 1-day lag
df_slow = ret.rolling(k_slow).mean().shift(1)

is_bull = (df_fast >= 0) & (df_slow >= 0)
is_bear = (df_fast < 0) & (df_slow < 0)
is_corr = (df_fast < 0) & (df_slow >= 0)
is_rebo = (df_fast >= 0) & (df_slow < 0)
is_else = ~(is_bull | is_bear | is_corr | is_rebo)

ret_fast = ret * (df_fast >= 0) + (-ret) * (df_fast < 0)
ret_slow = ret * (df_slow >= 0) + (-ret) * (df_slow < 0)


mean_bull = ret[is_bull].expanding().mean().reindex(ret.index).fillna(method='ffill')
mean_bear = ret[is_bear].expanding().mean().reindex(ret.index).fillna(method='ffill')
mean_corr = ret[is_corr].expanding().mean().reindex(ret.index).fillna(method='ffill')
mean_rebo = ret[is_rebo].expanding().mean().reindex(ret.index).fillna(method='ffill')

std_bull = ret[is_bull].expanding().std().reindex(ret.index).fillna(method='ffill')
std_bear = ret[is_bear].expanding().std().reindex(ret.index).fillna(method='ffill')
std_corr = ret[is_corr].expanding().std().reindex(ret.index).fillna(method='ffill')
std_rebo = ret[is_rebo].expanding().std().reindex(ret.index).fillna(method='ffill')
std_bube = ret[is_bull | is_bear].expanding().std().reindex(ret.index).fillna(method='ffill')

sharpe_bull = ((1 + mean_bull)**252 - 1) / (std_bull * np.sqrt(252))
sharpe_bear = ((1 + mean_bear)**252 - 1) / (std_bear * np.sqrt(252))
sharpe_corr = ((1 + mean_corr)**252 - 1) / (std_corr * np.sqrt(252))
sharpe_rebo = ((1 + mean_rebo)**252 - 1) / (std_rebo * np.sqrt(252))

freq_bull = is_bull.expanding().sum()
freq_bear = is_bear.expanding().sum()
freq_corr = is_corr.expanding().sum()
freq_rebo = is_rebo.expanding().sum()
freq_bube = (is_bull | is_bear).expanding().sum()

c = (freq_bull / freq_bube) * (mean_bull / std_bube) - (freq_bear / freq_bube) * (mean_bear / std_bube)
aco = 0.5 * (1 - (1 / c) * mean_corr / std_corr)
aco = np.minimum(aco, 1)
aco = np.maximum(aco, 0)

are = 0.5 * (1 - (1 / c) * mean_rebo / std_rebo)
are = np.minimum(are, 1)
are = np.maximum(are, 0)

ret_dyn = is_bull * ret + is_bear * (-ret) + is_corr * ((1 - aco) * ret_slow + aco * ret_fast) + is_rebo * ((1 - are) * ret_slow + are * ret_fast)
ret_dyn = ret_dyn.fillna(0)
dyn = (1 + ret_dyn).cumprod()

df = pd.concat([df, dyn.rename('DynTrend')], axis=1)
df = df[df.index >= '2017-01-01']
df = 100 * df / df.iloc[0]

df.plot()
plt.show()

ratio = df['DynTrend'] / df['IVVB11 BZ Equity']
ratio.plot()
plt.show()
