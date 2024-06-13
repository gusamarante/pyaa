from utils.stats import cov2corr
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from scipy.stats import norm
from hmmlearn import hmm  # TODO install this
from colour import Color
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np


class GaussianHMM:
    # TODO Documentation
    #      - instability of the estimator

    # TODO plot methods
    #      - networkx de transição de states
    #      - Forecast of states (based on current and on probabilities)
    #      - Simulate returns

    score = None
    aic = None
    bic = None
    trans_mat = None
    n_regimes = None
    avg_duration = None
    stationary_dist = None
    means = None
    vols = None
    covars = None
    corrs = None
    predicted_state = None
    state_freq = None
    state_probs = None

    def __init__(self, returns):
        self.returns = returns
        self.n_var = returns.shape[1]

    def select_order(self, max_regimes=8, select_iter=100, show_chart=False):
        # TODO Documentation
        df_select = pd.DataFrame()
        ns_range = range(2, max_regimes + 1)
        for ns in tqdm(ns_range, 'Computing scores for different number of regimes'):
            model = hmm.GaussianHMM(n_components=ns,
                                    covariance_type='full',
                                    n_iter=select_iter)
            model.fit(self.returns)
            df_select.loc[ns, "LogL"] = model.score(self.returns)
            df_select.loc[ns, "AIC"] = model.aic(self.returns)
            df_select.loc[ns, "BIC"] = model.bic(self.returns)

        if show_chart:
            fig, ax = plt.subplots()
            ln1 = ax.plot(ns_range, df_select['AIC'], label="AIC", color="blue", marker="o")
            ln2 = ax.plot(ns_range, df_select['BIC'], label="BIC", color="green", marker="o")
            ax2 = ax.twinx()
            ln3 = ax2.plot(ns_range, df_select['LogL'], label="LogL", color="orange", marker="o")

            ax.legend(handles=ax.lines + ax2.lines)
            ax.set_title("Using AIC/BIC for Model Selection")
            ax.set_ylabel("Criterion Value (lower is better)")
            ax2.set_ylabel("LL (higher is better)")
            ax.set_xlabel("Number of HMM Components")
            fig.tight_layout()
            plt.show()

        ll = df_select.idxmax()['LogL']
        aic = df_select.idxmin()['AIC']
        bic = df_select.idxmin()['BIC']

        return round((ll + aic + bic) / 3)

    def fit(self, n_regimes=None, select_iter=100):

        if n_regimes is None:
            n_regimes = self.select_order(
                show_chart=False,
                max_regimes=6,  # Forces a parsimonious choice
                select_iter=1000,
            )

        self.n_regimes = n_regimes
        model = hmm.GaussianHMM(n_components=n_regimes,
                                covariance_type='full',
                                n_iter=select_iter)
        model.fit(self.returns)
        sort_order = np.flip(np.argsort(np.diag(model.transmat_)))
        sorted_model = hmm.GaussianHMM(n_components=n_regimes,
                                       covariance_type='full')

        sorted_model.startprob_ = model.startprob_[sort_order]
        sorted_model.transmat_ = pd.DataFrame(model.transmat_).loc[sort_order, sort_order].values
        sorted_model.means_ = model.means_[sort_order, :]
        sorted_model.covars_ = model.covars_[sort_order, :, :]

        self.score = sorted_model.score(self.returns)
        self.aic = sorted_model.aic(self.returns)
        self.bic = sorted_model.bic(self.returns)
        self.trans_mat = pd.DataFrame(
            data=sorted_model.transmat_,
            index=[f'From State {s + 1}' for s in range(self.n_regimes)],
            columns=[f'To State {s + 1}' for s in range(self.n_regimes)],
        )
        self.avg_duration = pd.Series(
            data=1 / (1 - np.diag(sorted_model.transmat_)),
            index=[f'State {s + 1}' for s in range(self.n_regimes)],
            name='Average Duration',
        )
        self.stationary_dist = pd.Series(
            data=sorted_model.get_stationary_distribution(),
            index=[f'State {s + 1}' for s in range(self.n_regimes)],
            name='Stationary Distribution of States',
        )
        self.means = pd.DataFrame(
            data=sorted_model.means_,
            index=[f'State {s + 1}' for s in range(self.n_regimes)],
            columns=self.returns.columns,
        )
        vol_data = [list(np.sqrt(np.diag(sorted_model.covars_[ss]))) for ss in range(self.n_regimes)]
        self.vols = pd.DataFrame(
            data=vol_data,
            columns=self.returns.columns,
            index=[f'State {s + 1}' for s in range(self.n_regimes)],
        )
        idx = pd.MultiIndex.from_product(
            [
                [f'State {s + 1}' for s in range(self.n_regimes)],
                self.returns.columns,
            ]
        )
        self.covars = pd.DataFrame(
            index=idx,
            columns=self.returns.columns,
            data=sorted_model.covars_.reshape(-1, self.n_var),
        )
        corr_data = [cov2corr(sorted_model.covars_[ss])[0] for ss in range(self.n_regimes)]
        self.corrs = pd.DataFrame(
            index=idx,
            columns=self.returns.columns,
            data=np.concatenate(corr_data),
        )
        self.predicted_state = pd.Series(
            data=sorted_model.predict(self.returns) + 1,
            index=self.returns.index,
            name='Predicted State',
        )
        freq_data = ('State ' + self.predicted_state.astype(str)).value_counts() / self.predicted_state.count()
        self.state_freq = pd.Series(
            data=freq_data,
            index=[f'State {s + 1}' for s in range(self.n_regimes)],
            name='State Frequency',
        )
        self.state_probs = pd.DataFrame(
            data=sorted_model.predict_proba(self.returns),
            index=self.returns.index,
            columns=[f'State {s + 1}' for s in range(self.n_regimes)])


class GaussianHMMOld:

    def __init__(self, returns):
        self.returns = returns
        self.n_var = returns.shape[1]

    def select_order(self, max_state_number=8, select_iter=10, show_chart=False):
        self.state_selection = pd.Series(name='HMM Score')

        for ns in tqdm(range(1, max_state_number + 1), 'Computing scores for a different number of states'):
            max_score = - np.inf

            for _ in range(select_iter):
                model = hmm.GaussianHMM(n_components=ns,
                                        covariance_type='full',
                                        n_iter=1000)
                model.fit(self.returns)
                new_score = model.score(self.returns)
                if new_score > max_score:
                    max_score = new_score

            self.state_selection.loc[ns] = max_score

        most_concave = self.state_selection.diff().diff().idxmin()

        if show_chart:
            ax = self.state_selection.plot()
            loc = plticker.MultipleLocator(base=1)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Number of States')
            ax.set_ylabel('Log-likelihood')
            plt.tight_layout()
            plt.show()

        return most_concave

    def fit(self, fit_iter=100, n_states=None, max_state_number=8, select_iter=10):

        if n_states is None:
            self.n_states = self.select_order(max_state_number=max_state_number,
                                              select_iter=select_iter,
                                              show_chart=False)
        else:
            self.n_states = n_states

        # Estimate the model several times, due to instability, and grab the one with the highest score.
        model_dict = dict()
        for _ in tqdm(range(fit_iter), 'Estimating HMM'):
            model = hmm.GaussianHMM(n_components=self.n_states,
                                    covariance_type='full',
                                    n_iter=1000)
            model.fit(self.returns)
            model_dict[model.score(self.returns)] = model

        chosen_model = model_dict[max(model_dict.keys())]
        sort_order = np.flip(np.argsort(np.diag(chosen_model.transmat_)))

        # Build the sorted model
        sorted_model = hmm.GaussianHMM(n_components=self.n_states,
                                       covariance_type='full')

        sorted_model.startprob_ = chosen_model.startprob_[sort_order]
        sorted_model.transmat_ = pd.DataFrame(chosen_model.transmat_).loc[sort_order, sort_order].values
        sorted_model.means_ = chosen_model.means_[sort_order, :]
        sorted_model.covars_ = chosen_model.covars_[sort_order, :, :]

        try:
            column_labels = self.returns.columns
            time_index = self.returns.index
        except AttributeError:
            column_labels = [f'Asset {s + 1}' for s in range(self.n_var)]
            time_index = range(self.returns.shape[0])

        self.score = sorted_model.score(self.returns)

        self.trans_mat = pd.DataFrame(data=sorted_model.transmat_,
                                      index=[f'From State {s + 1}' for s in range(self.n_states)],
                                      columns=[f'To State {s + 1}' for s in range(self.n_states)])

        self.avg_duration = pd.Series(data=1 / (1 - np.diag(sorted_model.transmat_)),
                                      index=[f'State {s + 1}' for s in range(self.n_states)],
                                      name='Average Duration')

        self.stationary_dist = pd.Series(data=sorted_model.get_stationary_distribution(),
                                         index=[f'State {s + 1}' for s in range(self.n_states)],
                                         name='Stationary Distribution of States')

        self.means = pd.DataFrame(data=sorted_model.means_,
                                  index=[f'State {s + 1}' for s in range(self.n_states)],
                                  columns=column_labels)

        vol_data = [list(np.sqrt(np.diag(sorted_model.covars_[ss]))) for ss in range(self.n_states)]
        self.vols = pd.DataFrame(data=vol_data, columns=column_labels,
                                 index=[f'State {s + 1}' for s in range(self.n_states)])

        idx = pd.MultiIndex.from_product([[f'State {s + 1}' for s in range(self.n_states)],
                                          column_labels])
        self.covars = pd.DataFrame(index=idx, columns=column_labels,
                                   data=sorted_model.covars_.reshape(-1, self.n_var))

        corr_data = [cov2corr(sorted_model.covars_[ss])[0] for ss in range(self.n_states)]
        self.corrs = pd.DataFrame(index=idx, columns=column_labels,
                                  data=np.concatenate(corr_data))

        self.predicted_state = pd.Series(data=sorted_model.predict(self.returns) + 1,
                                         index=time_index,
                                         name='Predicted State')

        freq_data = ('State ' + self.predicted_state.astype(str)).value_counts() / self.predicted_state.count()
        self.state_freq = pd.Series(data=freq_data,
                                    index=[f'State {s + 1}' for s in range(self.n_states)],
                                    name='State Frequency')

        self.state_probs = pd.DataFrame(data=sorted_model.predict_proba(self.returns),
                                        index=time_index,
                                        columns=[f'State {s + 1}' for s in range(self.n_states)])

    def plot_series(self, data):
        white = Color("white")
        red = Color("red")
        colors = list(white.range_to(red, self.n_states))

        mindt, maxdt = min(self.predicted_state.index), max(self.predicted_state.index)
        data = data[data.index >= mindt]
        data = data[data.index <= maxdt]

        if isinstance(data, pd.Series):

            ax = data.plot(title=data.name)

            for st in range(self.n_states):
                dates = self.predicted_state[self.predicted_state == st + 1].index
                for dt in dates:
                    ax.axvspan(dt - pd.tseries.offsets.MonthBegin(),
                               dt + pd.tseries.offsets.MonthEnd(),
                               alpha=0.3, color=colors[st].hex, lw=0)

        else:
            # TODO support for DataFrames
            pass

        plt.tight_layout()
        plt.show()

    def plot_densities(self):

        n_subplots = self.n_var
        n_rows = int(np.floor(np.sqrt(n_subplots)))
        n_cols = int(np.ceil(n_subplots / n_rows))
        n_bins = int(np.ceil(np.sqrt(self.returns.shape[0])))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))

        for ax, asset in zip(axes.ravel(), list(self.returns.columns)):
            ax.set_title(asset)

            ax.hist(self.returns[asset], bins=n_bins, density=True, color='grey', alpha=0.3)
            xmin, xmax = ax.get_xlim()
            rangex = np.linspace(xmin, xmax, 100)
            mix_density = np.zeros(100)

            for state in range(self.n_states):
                mean = self.means[asset].iloc[state]
                std = self.vols[asset].iloc[state]
                density = self.stationary_dist.iloc[state] * norm(loc=mean, scale=std).pdf(rangex)
                mix_density = mix_density + density
                ax.plot(rangex, density, label=f'State {state + 1}', lw=1)

            ax.plot(rangex, mix_density, label='Mixture', lw=2)

        axes[0, 0].legend(loc='best')

        plt.tight_layout()
        plt.show()