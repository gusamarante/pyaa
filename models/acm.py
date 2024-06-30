import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import inv
from tqdm import tqdm


class NominalACM(object):
    """
    This class intends to replicate the paper 'pricing term structures with
    linear regressions' by Adrian, Crump & Moench (2013). It takes the zero
    curve vertices and excess returns of positions from all of them and return
    the term premium and risk neutral yields as object attributes.
    """

    base_count_dict = {'daily': 252,
                       'monthly': 12,
                       'yearly': 1}

    def __init__(self, curve, excess_returns, freq_mat='yearly', freq_obs='daily', interpolation='pchip', n_factors=5, compute_miy=False, verbose=False):
        """
        This class intends to replicate the paper 'pricing term structures with
        linear regressions' by Adrian, Crump & Moench (2013). It takes the zero
        curve vertices and excess returns of positions from all of them and
        return the term premium and risk neutral yields as object attributes.

        Parameters
        ----------
        curve : pd.Dataframe
            Equally spaced vertices as columns. Values must be in percent, not
            percentage points

        excess_returns : pd.DataFrame
            the excess returns of the vertices as columns

        freq_mat : str
            Space between each observed maturity

        freq_obs : str
            Time between each observation

        # TODO what to do here?
        """

        # TODO assert curve and excess returns have the same index
        # TODO assert curve and excess returns have the same columns
        # TODO normalize all array operations to have 2 dimensions
        self.curve = curve
        self.excess_returns = excess_returns
        self.curve_exp = np.log(1 + curve)
        self.n_factors = n_factors
        self.n_tenors = excess_returns.shape[1]
        self.tenors = excess_returns.columns  # TODO Remove this
        self.sample_size = curve.shape[0]
        self.base_count = self.base_count_dict[freq_mat]
        self.periods_in_year = self.base_count_dict[freq_obs]
        self.compute_miy = compute_miy

        self._run_estimation()

    def _run_estimation(self):

        # Step 0 - get the PCA factor series of the yield curve
        self.pca_factors, self.pca_loadings = self._get_pca_factors()

        # Step 1 - VAR for the PCA factors
        Mu_hat, Phi_hat, V_hat, Sigma_hat = self._estimate_factor_var()

        # Step 2 - Excess return equation
        beta_hat, a_hat, B_star_hat, sigma2_hat, c_hat = self._estimate_excess_return_equation(v_hat=V_hat)

        # Step 3 - Estimate price of risk parameters
        lambda_0_hat, lambda_1_hat = self._retrieve_lambda(beta_hat, a_hat, B_star_hat, Sigma_hat, sigma2_hat, c_hat)

        # Step 4 - Equation for the Short Rate
        delta_0_hat, delta_1_hat = self._estimate_short_rate_equation()

        # Step 5 - Affine Recursions
        # model implied yield
        if self.compute_miy:
            miy = self._affine_recursions(Mu_hat, Phi_hat, Sigma_hat, sigma2_hat, lambda_0_hat, lambda_1_hat,
                                          delta_0_hat, delta_1_hat)

            miy = pd.DataFrame(data=miy[:, 1:],
                               index=self.pca_factors.index,
                               columns=[i + 1 for i in range(self.tenors.max())])

            self.miy = np.exp(miy) - 1
        else:
            self.miy = None

        # risk neutral yield
        rny = self._affine_recursions(Mu_hat, Phi_hat, Sigma_hat, sigma2_hat, 0, 0, delta_0_hat, delta_1_hat)

        rny = pd.DataFrame(data=rny[:, 1:],
                           index=self.PCA_factors[1:].index,
                           columns=list(range(1, self.tenors.max() + 1)))

        self.rny = np.exp(rny) - 1
        self.term_premium = ((1 + self.curve) / (1 + self.rny) - 1).dropna(how='all')

    def _get_pca_factors(self):

        pca = PCA(n_components=self.n_factors)

        col_names = [f'PC {i+1}' for i in range(self.n_factors)]
        df_pca = pd.DataFrame(data=pca.fit_transform(self.curve_exp),
                              index=self.curve.index,
                              columns=col_names)

        df_loadings = pd.DataFrame(data=pca.components_.T,
                                   columns=col_names,
                                   index=self.curve.columns)

        # Standardize the sign
        signal = np.sign(df_loadings.iloc[-1])
        df_loadings = df_loadings * signal
        df_pca = df_pca * signal

        return df_pca, df_loadings

    def _estimate_factor_var(self):
        Y = self.pca_factors.iloc[1:].copy()
        Z = self.pca_factors.iloc[:-1].copy()
        Z.insert(0, 'const', 1)
        Z = Z.T

        # VAR(1) estimator is given by equation (3.2.10) from Lutkepohl's book.
        mat_Z = Z.values
        mat_Y = Y.values.T
        B_hat = mat_Y @ (mat_Z.T @ inv(mat_Z @ mat_Z.T))

        # Computes matrices Mu and Phi of the VAR(1) of the paper.
        Mu_hat = B_hat[:, 0]
        Phi_hat = B_hat[:, 1:]

        # residuals matrix V_hat and the unbiased estimate of its covariance
        V_hat = mat_Y - (B_hat @ mat_Z)
        Sigma_hat = (1 / (self.sample_size - self.n_factors - 1)) * (V_hat @ V_hat.T)

        # Convert frequency of the parameters from observations to maturities
        # TODO Parei aqui

        return Mu_hat, Phi_hat, V_hat, Sigma_hat

    def _estimate_excess_return_equation(self, v_hat):

        mat_rx = self.excess_returns.iloc[1:].values.T

        Z = np.concatenate((np.ones((1, self.sample_size - 1)), v_hat, self.pca_factors.iloc[:-1].T))

        D_hat = mat_rx @ (Z.T @ inv(Z @ Z.T))
        a_hat = D_hat[:, 0]
        beta_hat = D_hat[:, 1:self.n_factors + 1].T
        c_hat = D_hat[:, self.n_factors + 1:]

        E_hat = mat_rx - (D_hat @ Z)
        sigma2_hat = np.trace(E_hat @ E_hat.T) / (self.n_tenors * (self.sample_size - 1))

        # Builds the B* matrix, defined in equation (13) of the paper
        B_star_hat = np.zeros((self.n_tenors, self.n_factors ** 2))
        for i in range(0, self.n_tenors):
            beta_col = beta_hat[:, i].reshape((-1, 1))
            B_star_hat[i, :] = np.reshape(beta_col @ beta_col.T, (1, self.n_factors ** 2))

        # beta_hat = np.array(beta_hat)  # TODO remove this?

        return beta_hat, a_hat, B_star_hat, sigma2_hat, c_hat

    def _retrieve_lambda(self, beta_hat, a_hat, b_star_hat, Sigma_hat, sigma2_hat, c_hat):

        # beta_hat = np.matrix(beta_hat)  # TODO remove these?
        # a_hat = np.matrix(a_hat)
        a_hat = a_hat.reshape((-1, 1))
        # b_star_hat = np.matrix(b_star_hat)
        # Sigma_hat = np.matrix(Sigma_hat)
        # c_hat = np.matrix(c_hat)

        vecSigma = Sigma_hat.reshape((-1, 1))
        lambda_0_hat = inv(beta_hat @ beta_hat.T) @ (beta_hat @ (a_hat + 0.5 * (b_star_hat @ vecSigma + sigma2_hat * np.ones((self.n_tenors, 1)))))
        lambda_1_hat = (inv(beta_hat @ beta_hat.T) @ beta_hat) @ c_hat

        return lambda_0_hat, lambda_1_hat

    def _estimate_short_rate_equation(self):

        X_star = self.pca_factors.copy()
        X_star.insert(0, 'const', 1)
        X_star = X_star.values

        r1 = ((1/self.base_count) * self.curve_exp.iloc[:, 0].values).reshape((-1, 1))

        # Delta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X_star.T, X_star)), X_star.T), r1)  # TODO remove this
        Delta_hat = inv(X_star.T @ X_star) @ X_star.T @ r1
        delta_0_hat = Delta_hat[0]
        delta_1_hat = Delta_hat[1:]

        return delta_0_hat, delta_1_hat

    def _affine_recursions(self, Mu_hat, Phi_hat, Sigma_hat, sigma2_hat, lambda_0_hat, lambda_1_hat, delta_0_hat,
                           delta_1_hat):

        Mu_hat = Mu_hat.reshape((-1, 1))

        X_star = self.pca_factors.copy()
        # X_star.insert(0, 'const', 1)
        X_star = X_star.values

        N_rec, N_start = self.tenors.max(), self.tenors.min()

        Bn = np.zeros((self.n_factors, N_rec + 1))
        # Bn[:, 1] = - delta_1_hat.reshape((self.n_factors, ))

        for i in range(N_start, N_rec + 1):
            Bn[:, i] = (Bn[:, i - 1].T @ (Phi_hat - lambda_1_hat) - delta_1_hat.T).T.reshape((-1, ))

        An = np.zeros((1, N_rec + 1))
        # An[:, 1] = -delta_0_hat

        for i in range(N_start, N_rec + 1):
            An[:, i] = An[:, i - 1] + Bn[:, i - 1].T @ (Mu_hat - lambda_0_hat) + 0.5 * (Bn[:, i - 1].T @ Sigma_hat @ Bn[:, i - 1] + sigma2_hat) - delta_0_hat

        miy = np.zeros((self.sample_size, N_rec + 1))
        Xt = X_star.T

        for t in tqdm(range(self.sample_size), 'Computing Yields'):
            for n in range(N_start, N_rec + 1):  # iterate on the maturities
                miy[t, n] = - (self.base_count / n) * (An[:, n] + Bn[:, n].T @ Xt[:, t])

        return miy
