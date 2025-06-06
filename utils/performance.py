import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


class Performance(object):

    # TODO Review
    # TODO Decompose returns seasonally by month/year

    def __init__(self, total_return, rolling_window=252, skip_dd=False, freq="D"):
        """
        Computes performance measures for each column of 'total_return'

        Parameters
        ----------
        total_return: pandas.DataFrame
            Total return inndexes

        rolling_window: int
            number of business day for the rolling measures
        """
        # TODO better documentation

        assert isinstance(total_return, pd.DataFrame), \
            "'total_return' must be a pandas DataFrame, even if there is only one column"

        self.freq = freq

        # TODO better implementation
        if self.freq == "D":
            self.ret_factor = 252
        elif self.freq == "M":
            self.ret_factor = 12
        else:
            raise NotImplementedError("frequancy not implemented")

        self.total_return = total_return
        self.start_date = total_return.isna().astype(int).diff().idxmin().rename('Start Date')
        self.rolling_window = rolling_window
        self.returns_ts = total_return.pct_change(1)
        self.returns_ann = self._get_ann_returns()
        self.std = self._get_ann_std()
        self.sharpe = self.returns_ann / self.std
        self.skewness = self.returns_ts.skew()
        self.kurtosis = self.returns_ts.kurt()  # Excess Kurtosis (k=0 is normal)
        self.sortino = self._get_sortino()

        if skip_dd:
            self.drawdowns = None
            self.max_dd = (total_return / total_return.expanding().max()).min() - 1
        else:
            self.drawdowns = self._get_drawdowns()
            self.max_dd = self.drawdowns.groupby(level=0).min()['dd']

        self.table = self._get_perf_table(skip_dd=skip_dd)
        self.rolling_return = (1 + self.total_return.pct_change(rolling_window)) ** (252 / rolling_window) - 1
        self.rolling_std = self.returns_ts.rolling(rolling_window).std() * np.sqrt(252)
        self.rolling_sharpe = self.rolling_return / self.rolling_std
        self.rolling_skewness = self.returns_ts.rolling(rolling_window).skew()
        self.rolling_kurtosis = self.returns_ts.rolling(rolling_window).kurt()

    def plot_drawdowns(self, name, n=5, show_chart=False, save_path=None):
        # TODO Documentation

        tri = self.total_return[name].interpolate(limit_area='inside')
        fig = plt.figure(figsize=(12, 12 * 0.61))
        ax = fig.gca()
        plt.plot(tri, color='#0000CD', linewidth=1)

        for dd in range(n):
            try:
                start = self.drawdowns.loc[name].loc[dd, 'start']
                end = self.drawdowns.loc[name].loc[dd, 'end']
                plt.plot(tri.loc[start: end], color='#E00000')
            except KeyError:
                pass

        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
        plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

        plt.xticks(rotation=90)
        locators = mdates.YearLocator()
        ax.xaxis.set_major_locator(locators)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)

        ax.set_title(f'{name} - {n} Biggest Drawdowns', fontdict={'fontsize': 15 + 2, 'fontweight': 'bold'})

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

    def plot_underwater(self, name, show_chart=False, save_path=None):
        # TODO Documentation

        tr = self.total_return[name].dropna()
        exp_max = tr.expanding().max()
        uw = 100 * (tr / exp_max - 1)

        fig = plt.figure(figsize=(12, 12 * 0.61))
        ax = fig.gca()
        ax.plot(uw, color='#0000CD', linewidth=1)
        ax.axhline(0, color='black', linewidth=1)

        # Plot percentiles
        props = dict(boxstyle='round', facecolor='white', alpha=1)

        q10 = self.drawdowns.loc[name, 'dd'].quantile(0.10) * 100
        q5 = self.drawdowns.loc[name, 'dd'].quantile(0.05) * 100
        q1 = self.drawdowns.loc[name, 'dd'].quantile(0.01) * 100

        ax.axhline(q10, color='#E00000', linewidth=1)
        ax.text(uw.index[0], q10, '10% Percentile', color='#E00000', size=13,
                bbox=props, verticalalignment='top', horizontalalignment='left')

        ax.axhline(q5, color='#E00000', linewidth=1)
        ax.text(uw.index[0], q5, '5% Percentile', color='#E00000', size=13,
                bbox=props, verticalalignment='top', horizontalalignment='left')

        ax.axhline(q1, color='#E00000', linewidth=1)
        ax.text(uw.index[0], q1, '1% Percentile', color='#E00000', size=13,
                bbox=props, verticalalignment='top', horizontalalignment='left')

        ax.set_title(f'{name} - Underwater Chart',
                     fontdict={'fontsize': 15 + 2, 'fontweight': 'bold'})

        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
        plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

        plt.xticks(rotation=90)
        locators = mdates.YearLocator()
        ax.xaxis.set_major_locator(locators)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax.yaxis.set_major_formatter(PercentFormatter())

        ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

    def _get_drawdowns(self):
        """
        Finds all the drawdowns, from peak to bottom, and its dates. At the
        margin, even if drawdown is not recovered, it treats the local bottom as
        the bottom of the last drawdon.
        :return: pandas.DataFrame with a MultiIndex. The first level is the asset,
                 and the second are the ranked drawdowns.
        """

        df_drawdowns = pd.DataFrame()

        for col in tqdm(self.total_return.columns, 'Computing Drawdowns'):
            data = pd.DataFrame(index=self.total_return[col].dropna().index)
            data['tracker'] = self.total_return[col].dropna()
            data['expanding max'] = data['tracker'].expanding().max()
            data['dd'] = data['tracker'] / data['expanding max'] - 1
            data['iszero'] = data['dd'] == 0
            data['current min'] = 0.0
            data['last start'] = data.index[0]
            data['end'] = data.index[0]

            # Find the rolling current worst drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'iszero']:
                    data.loc[date, 'current min'] = 0
                elif data.loc[date, 'dd'] < data.loc[datem1, 'current min']:
                    data.loc[date, 'current min'] = data.loc[date, 'dd']
                else:
                    data.loc[date, 'current min'] = data.loc[datem1, 'current min']

            # find the last start of the drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'iszero']:
                    data.loc[date, 'last start'] = date
                else:
                    data.loc[date, 'last start'] = data.loc[datem1, 'last start']

            # find the end of each drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'current min'] < data.loc[datem1, 'current min']:
                    data.loc[date, 'end'] = date
                else:
                    data.loc[date, 'end'] = data.loc[datem1, 'end']

            # find drawdown splits
            data['dd duration'] = (data['end'] - data['last start']).dt.days
            data['dd duration shift'] = data['dd duration'].shift(-1)
            data['isnegative'] = data['dd duration shift'] < 0
            data.iloc[-1, -1] = True

            data = data.reset_index()

            data = data[data['isnegative']]

            data = data.sort_values('current min', ascending=True)

            data = data[data['current min'] < 0]

            data = data[['current min', 'last start', 'end']].reset_index().drop('index', axis=1)

            df_add = pd.DataFrame(index=pd.MultiIndex.from_product([[col], data.index]),
                                  data=data.values,
                                  columns=['dd', 'start', 'end'])

            df_add['duration'] = (df_add['end'] - df_add['start']).dt.days

            df_drawdowns = pd.concat([df_drawdowns, df_add])

        df_drawdowns['dd'] = df_drawdowns['dd'].astype(float)
        return df_drawdowns

    def _get_ann_returns(self):

        df_ret = pd.Series(name='Annualized Returns', dtype=float)

        for col in self.total_return.columns:
            aux = self.total_return[col].dropna()
            start, end = aux.iloc[0], aux.iloc[-1]
            n = len(aux) - 1
            df_ret.loc[col] = (end / start) ** (self.ret_factor / n) - 1

        return df_ret

    def _get_ann_std(self):

        df_std = pd.Series(name='Annualized Standard Deviation', dtype=float)
        for col in self.total_return.columns:
            aux = self.returns_ts[col].dropna()
            df_std.loc[col] = aux.std() * np.sqrt(self.ret_factor)

        return df_std

    def _get_sortino(self):

        df_sor = pd.Series(name='Sortino', dtype=float)
        for col in self.total_return.columns:
            aux = self.returns_ts[col][self.returns_ts[col] < 0].dropna()
            df_sor.loc[col] = self.returns_ann[col] / (np.sqrt(self.ret_factor) * aux.std())

        return df_sor

    def _get_perf_table(self, skip_dd=False):

        df = pd.DataFrame(columns=self.total_return.columns)

        df.loc['Return'] = self.returns_ann
        df.loc['Vol'] = self.std
        df.loc['Sharpe'] = self.sharpe
        df.loc['Skew'] = self.skewness
        df.loc['Kurt'] = self.kurtosis
        df.loc['Sortino'] = self.sortino
        df.loc['Max DD'] = self.max_dd

        if not skip_dd:
            df.loc['DD 5%q'] = self.drawdowns.reset_index().groupby('level_0').quantile(0.05)['dd']

        df.loc['Start Date'] = self.start_date

        return df


def compute_eri(total_return_index, funding_return):
    # TODO review
    """
    Computes the excess returns indexes based on several total return indexes and one series of
    funding returns (risk-free rate). For now, there are no resample adjustments, so both of the
    frequencies must be in their correct timing and measurement.
    :param total_return_index: pandas.DataFrame with the total return indexes
    :param funding_return: pandas.Series with the funding returns
    :return: pandas.DataFrame where each column is the Excess Return Index for that given total return index.
    """
    assert isinstance(funding_return, pd.Series), 'funding returns must be a pandas.Series'

    total_returns = total_return_index.pct_change(1, fill_method=None)
    er = total_returns.subtract(funding_return, axis=0)
    er = er.dropna(how='all')

    eri = (1 + er).cumprod()
    eri = 100 * eri / eri.fillna(method='bfill').iloc[0]

    return eri


def rescale_vol(df, target_vol=0.1):
    """
    Rescale return indexes (total or excess) to have the desired volatitlity.
    :param df: pandas.DataFrame of return indexes.
    :param target_vol: float with the desired volatility.
    :return: pandas.DataFrame with rescaled total return indexes.
    """
    returns = df.pct_change(1, fill_method=None)
    returns_std = returns.std() * np.sqrt(252)
    returns = target_vol * returns / returns_std

    df_adj = (1 + returns).cumprod()
    df_adj = 100 * df_adj / df_adj.fillna(method='bfill').iloc[0]

    return df_adj
