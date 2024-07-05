from models import NominalACM
from data import curve_di

di = curve_di()
di = di.loc[:, :'120m'].dropna()
print(di)
