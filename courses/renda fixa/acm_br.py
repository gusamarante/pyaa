import matplotlib.pyplot as plt
from models import NominalACM
from data import curve_di

di = curve_di()
di = di.loc[:, :'120m'].dropna()
di = di[di.index >= "2008-01-01"]

acm = NominalACM(
    curve=di,
)

mat = "60m"
di[mat].rename('Observed').plot(title="5y DI", legend=True)
acm.miy[mat].rename('Fitted').plot(legend=True)
acm.rny[mat].rename('Risk-Neutral').plot(legend=True)
plt.show()
