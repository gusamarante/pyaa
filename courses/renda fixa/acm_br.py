from models import NominalACM
from data import curve_di
import numpy as np

di = curve_di()
di = di.loc[:, :'120m'].dropna()
di = di[di.index >= "2008-01-01"]


# Convert yields to log-yields
# TODO should this go inside the class?
log_di = np.log(1 + di)


acm = NominalACM(
    curve=log_di,
)
