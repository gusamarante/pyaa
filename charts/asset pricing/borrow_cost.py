"""
Ilustratoin of the Mean Variance when there are borrowing costs
"""
from allocation import MeanVar
import pandas as pd
from utils import corr2cov

# Parameters
mu = pd.Series(index=['A', 'B', 'C'],
               data=[0.1, 0.15, 0.20])

sigma = pd.Series(index=['A', 'B', 'C'],
               data=[0.1, 0.15, 0.20])

corr = pd.DataFrame(columns=['A', 'B', 'C'],
                    index=['A', 'B', 'C'],
                    data=[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

cov = corr2cov(corr, sigma)

mv = MeanVar(mu, cov, rf=0.02, rb=0.09)
mv.plot(mvfnoss=False, investor=False, gmvp=False)
