import matplotlib.pyplot as plt
from models import NominalACM
from data import curve_di

di = curve_di()
di = di.loc[:, :'121m'].dropna()

acm = NominalACM(
    curve=di,
)

mat = "60m"
di[mat].rename('Observed').plot(title="5y DI", legend=True)
acm.miy[mat].rename('Fitted').plot(legend=True)
acm.rny[mat].rename('Risk-Neutral').plot(legend=True)
plt.show()

acm.er_hist_d[mat].plot()
plt.show()

cols2view = [f"{mat*6}m" for mat in range(1, 21)]
acm.er_hist_d.iloc[-1].loc[cols2view].plot(kind='bar')
plt.show()
