"""
This is very far from finished.
Supposed to be a bayesian cross validation, but there is still a lot to do

First I need a better example data set, where the penalization parameter has a higher effect on the CV process
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

def func(x):
    return np.sin(2 * np.pi * x)


# Ganerate and plot data
size = 25
rng = np.random.RandomState()
X = rng.uniform(0.0, 1.0, size)
y = func(X) + rng.normal(scale=0.1, size=size)

plt.scatter(X, y)
plt.show()

X_plot = np.linspace(0, 1, 100)

# Lasso with fixed alpha
lasso = linear_model.Lasso(alpha=0.00001, fit_intercept=False)
lasso.fit(np.vander(X, N=20), y)
pred = lasso.predict(np.vander(X_plot, N=20))

plt.scatter(X, y)
plt.plot(X_plot, pred)
plt.show()

# Set all alphas to be tested
n_alphas = 100
alphas = np.logspace(-5, 0, n_alphas)


# Lasso with cross validation
lasso = linear_model.LassoCV(alphas=alphas, fit_intercept=False, cv=5)
lasso.fit(np.vander(X, N=20), y)
print("chosen alpha", lasso.alpha_)

pred = lasso.predict(np.vander(X_plot, N=20))

plt.scatter(X, y)
plt.plot(X_plot, pred)
plt.show()


# See the parameters shirinking
coefs = []
for a in alphas:
    ridge = linear_model.Lasso(alpha=a, fit_intercept=False)
    ridge.fit(np.vander(X, N=20), y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()