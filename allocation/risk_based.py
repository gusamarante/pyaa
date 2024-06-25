import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
from utils.stats import cov2corr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np


class HRP:
    """
    Implements Hierarchical Risk Parity
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
    """

    def __init__(self, cov, method='single', metric='euclidean'):
        """
        Implements HRP allocation based on covariance matrix `cov`.
        This allows for the user to pass covariances matrices that
        have been treated in any desired way (detoned, denoised,
        shrunk, etc.). It even allows for singular covariances, which
        is comceptually wrong in asset pricing theory, but allowed in
        this allocation method.


        Parameters
        ----------
        cov : pandas.DataFrame
            Covaraince matrix of returns

        method : string
            The linkage algorithm to use. Takes in any method available
            in scipy.cluster.hierarchy.linkage

        metric : string
            The distance metric to use. Takes in any metric available
            in scipy.cluster.hierarchy.linkage


        Attributes
        ----------
        weights : pd.Series
            final HRP weights for each asset

        cov : pandas.DataFrame
            Covariance matrix of the returns

        corr : pandas.DataFrame
            Correlation matrix of the returns

        sorted_corr : pandas.DataFrame
            Correlation matrix of the returns, ordered according to the
            quasi-diagonalization step of the HRP algorithm

        sort_ix : list
            asdf

        link : numpy.ndarray
            linkage matrix of size (N-1)x4 with structure
                Y=[{y_m,1  y_m,2  y_m,3  y_m,4}_m=1,N-1]
            At the i-th iteration, clusters with indices link[i, 0] and
            link[i, 1] are combined to form cluster n+1. A cluster with
            an index less than n corresponds to one of the original
            observations.
            The distance between clusters link[i, 0] and link[i, 1] is
            given by link[i, 2]. The fourth value link[i, 3] represents
            the number of original observations in the newly formed cluster.
        """

        assert isinstance(cov, pd.DataFrame), "input 'cov' must be a pandas DataFrame"

        self.cov = cov
        self.corr, self.vols = cov2corr(cov)
        self.method = method
        self.metric = metric

        self.link = self._tree_clustering(self.corr, method, metric)
        sort_ix_numbers = self._get_quasi_diag(self.link)
        self.sort_ix = self.corr.index[sort_ix_numbers].tolist()  # recover labels
        self.sorted_corr = self.corr.loc[self.sort_ix, self.sort_ix]  # reorder correlation matrix
        self.weights = self._get_recursive_bisection(self.cov, self.sort_ix)

    @staticmethod
    def _tree_clustering(corr, method, metric):
        dist = np.sqrt((1 - corr)/2)
        link = sch.linkage(dist, method, metric)
        return link

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = pd.concat([sort_ix, df0])  # item 2  # TODO change append to concat
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    def _get_recursive_bisection(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix, name='HRP')
        c_items = [sort_ix]  # initialize all items in one cluster
        # c_items = sort_ix

        while len(c_items) > 0:

            # bi-section
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]

            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i + 1]  # cluster 2
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1 - alpha  # weight 2
        return w

    def _get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]  # matrix slice
        w_ = self._get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def _get_ivp(cov):
        ivp = 1 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def plot_corr_matrix(self, save_path=None, show_chart=True, cmap='vlag', figsize=(7, 7)):
        """
        Plots the correlation matrix

        Parameters
        ----------
        save_path : str or Path
            local directory to save file. If provided, saves the image to the address

        show_chart: bool
            If True, shows the chart

        cmap: str
            matplotlib colormap for the heatmap of the correlation matrix

        figsize: tuple
            figsize dimensions
        """

        sns.clustermap(data=self.corr,
                       method=self.method,
                       metric=self.metric,
                       cmap=cmap,
                       figsize=figsize,
                       linewidths=0,
                       col_linkage=self.link,
                       row_linkage=self.link,
                       vmin=-1,
                       vmax=1)

        if not (save_path is None):
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

    def plot_dendrogram(self, show_chart=True, save_path=None, figsize=(7, 7), threshold=None):
        """
        Plots the dendrogram using scipy's own method.

        Parameters
        ----------
        show_chart : bool
            If True, shows the chart

        save_path : str or Path
            local directory to save file

        figsize: tuple
            figsize dimensions

        threshold: float
            height of the dendrogram to color the nodes. If None, the colors of the nodes
            follow scipy's standard behaviour, which cuts the dendrogram on 70% of its
            height (0.7 * max(self.link[:, 2]).
        """

        plt.figure(figsize=figsize)
        dn = sch.dendrogram(self.link, orientation='left', labels=self.corr.columns.to_list(),
                            color_threshold=threshold)

        plt.tight_layout()

        if not (save_path is None):
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()


class RiskBudgetVol:
    """
    Implements Risk Bedgeting Portfolio described in Thierry Roncalli's book,
    "Introduction to Risk Parity and Budgeting" (2014), using volatility as
    the risk measure of the portfolio
    """

    def __init__(self, cov, budget=None):
        # TODO assert that indexes of cov and budget match
        """
        Combines the assets in `cov` so that all of them have equal contributions
        to the overall volatility of the portfolio. The weights are bounded
        between 0 and 1, as suggested by the authors.

        Parameters
        ----------
        cov : pandas.DataFrame
            covariance matrix of returns

        budget : pandas.Series or None
            The risk budget vector, each entry is the share of risk contributions
            of the asset. This vector must add up to 1. If None is passed, equal
            risk contribution from each asset is assumed.
        """

        self.n_assets = cov.shape[0]
        self.cov = cov

        if budget is None:
            # Assume ERC
            self.budget = np.ones(self.n_assets) / self.n_assets
        else:
            self.budget = budget

        # --- Optimization ---
        bounds = np.hstack([np.zeros((self.n_assets, 1)), np.ones((self.n_assets, 1))])
        cons = ({'type': 'ineq',
                 'fun': lambda w: self._port_vol(w)},  # <= 0
                {'type': 'eq',
                 'fun': lambda w: 1 - w.sum()})
        res = minimize(fun=self._dist_to_target,
                       x0=self.budget,
                       method='SLSQP',
                       constraints=cons,
                       bounds=bounds)

        assert res.success, "Optimization did not converge"

        # --- Utilize optimization outputs ---
        self.weights = pd.Series(index=cov.columns, data=res.x, name='RB Vol')
        self.vol = np.sqrt(res.x @ cov @ res.x)
        self.marginal_risk = (res.x @ cov) / self.vol
        self.risk_contribution = self.marginal_risk * self.weights
        self.risk_contribution_ratio = self.risk_contribution / self.vol

    def _port_vol(self, w):
        return np.sqrt(w.dot(self.cov).dot(w))

    def _risk_contribution(self, w):
        return w * ((w @ self.cov) / (self._port_vol(w)**2))

    def _dist_to_target(self, w):
        return ((self._risk_contribution(w) - self.budget)**2).sum()


class VolTartget:

    def __init__(self, tracker, vol_method='sd ewm', vol_target=0.1):
        
        # Basic chacks
        msg = "`tracker` is not a pandas object"
        assert isinstance(tracker, (pd.Series, pd.DataFrame)), msg

        # Save inputs as attributes
        self.tracker = tracker
        self.vol_method = vol_method

        # ===== Backtest ===
        # Vol target / leverage factor
        daily_vol = self._get_daily_vol(tracker, vol_method)
        self.vol = daily_vol.copy()
        scaling = vol_target / daily_vol.shift(2)  # already lagged
        scaling = scaling.dropna()

        # auxiliary variables
        backtest = pd.Series()
        holdings = pd.Series()
        start_date = scaling.index[0]
        deltap = tracker.diff(1)
        deltap = deltap.fillna(0)

        # --- Initial Date ---
        holdings.loc[start_date] = scaling.loc[start_date]
        backtest.loc[start_date] = 100
        next_rebal = start_date + pd.offsets.DateOffset(months=1)

        # --- loop other dates ---
        dates2loop = zip(scaling.index[1:], scaling.index[:-1])
        for d, dm1 in tqdm(dates2loop):

            pnl = holdings.loc[dm1] * deltap.loc[d]
            backtest.loc[d] = backtest.loc[dm1] + pnl

            if d >= next_rebal:
                holdings.loc[d] = scaling.loc[d]
                next_rebal = d + pd.offsets.DateOffset(months=1)
            else:
                holdings.loc[d] = holdings.loc[dm1]

        self.holdings = holdings.dropna()
        self.backtest = backtest.dropna()

    def risk_return_tradeoff(self, n_bins=5):
        """
        Replicates the charts for Risk-return tradeoff heterogheneity found in
        the vol targeting studies

        Parameters
        __________
        n_bins : int
            Number of bins in the percentile groupings
        """
        vol = self.vol.resample('M').last()
        vol_lag = vol.shift(1)
        ret = self.tracker.resample('M').last().pct_change(1) * 12

        qtiles = pd.qcut(vol_lag, q=n_bins, labels=False)

        df = pd.concat(
            [
                ret.rename('Returns'),
                vol.rename('Vol'),
                qtiles.rename('Quantile') + 1,
            ],
            axis=1,
        )
        df = df.groupby('Quantile').mean()
        df['Sharpe'] = df['Returns'] / df['Vol']

        # --- Chart ---
        fig = plt.figure(figsize=(6 * (16 / 7.3), 6))

        # Returns
        ax = plt.subplot2grid((1, 3), (0, 0))
        ax.bar(df.index, df['Returns'])
        ax.set_xlabel("Percentile Group of Previous Month's Vol")
        ax.set_ylabel("Annualized Return of the Following Month")
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

        # Vol
        ax = plt.subplot2grid((1, 3), (0, 1))
        ax.bar(df.index, df['Vol'])
        ax.set_xlabel("Percentile Group of Previous Month's Vol")
        ax.set_ylabel("Annualized Vol of the Following Month")
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

        # Sharpe
        ax = plt.subplot2grid((1, 3), (0, 2))
        ax.bar(df.index, df['Sharpe'])
        ax.set_xlabel("Percentile Group of Previous Month's Vol")
        ax.set_ylabel("Sharpe of the Following Month")
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def _get_daily_vol(tracker, vol_method):
        # Different Vol estilmation methods go here

        if vol_method == 'sd ewm':
            returns = tracker.pct_change(1)
            vols = returns.ewm(com=21).std()
            vols = vols * np.sqrt(252)
        else:
            raise NotImplementedError(f"vol method {vol_method} not available")

        return vols

