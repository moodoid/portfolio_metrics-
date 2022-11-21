import pandas as pd
import numpy as np
import datetime as dt

from typing import NoReturn
from value_at_risk import ValueAtRisk


def monthlist_fast(dates):
    start, end = dates[0], dates[1]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start) - 1, total_months(end) + 1):
        y, m = divmod(tot_m, 12)
        mlist.append(dt.datetime(y, m + 1, 1).strftime("%Y-%m"))
    return mlist


def handle_benchmark_comparison(func):
    def _apply_handle(self, *args, **kwargs):
        # assert that the frequency of the portfolio price series and the benchmark are the same
        # assert pd.infer_freq(getattr(self, 'prices').index) == pd.infer_freq(getattr(self, 'benchmark').index)

        # assert that there are common dates for applicable timeseries comparison
        assert not self.returns.index.intersection(self.benchmark.index).empty

        return func(self, *args, **kwargs)

    return _apply_handle


class PortfolioMetrics(ValueAtRisk):
    def __init__(self, portfolio: pd.Series, benchmark: pd.Series = None, rfr: float = 0.0, ann_factor: int = 252):
        super().__init__(data=portfolio, alpha=.05, smooth_factor=.95, pct=True)

        assert isinstance(portfolio, pd.Series)
        self.check_dtx(frame_=portfolio)

        portfolio = portfolio.resample('1d').last()

        if not isinstance(benchmark, pd.Series):
            self.benchmark = pd.DataFrame()
        else:
            self.check_dtx(frame_=benchmark)
            self.benchmark = benchmark

        self.portfolio = portfolio
        self.rfr = rfr

        self._ann_factor = ann_factor

    @staticmethod
    def check_dtx(frame_: pd.Series) -> NoReturn:
        if isinstance(frame_.index, pd.DatetimeIndex):
            return
        else:
            raise Exception('Please provide a pd.DatetimeIndex')

    def get_ann_return(self, data: pd.Series) -> float:
        if self._ann_factor < 252:
            days_range = np.busday_count(data.index.min().strftime('%Y-%m-%d'), data.index.max().strftime('%Y-%m-%d'))
        else:
            days_range = (data.index.max() - data.index.min()).days

        a_return = (data.iloc[-1] / data.iloc[0]) ** (self._ann_factor / days_range) - 1

        return a_return

    @property
    def prices(self) -> pd.Series:
        return self.portfolio

    @property
    def returns(self) -> pd.Series:
        return self.prices.pct_change().dropna()

    @property
    def ann_return(self) -> float:
        return self.get_ann_return(data=self.prices)

    @property
    def ann_volatility(self) -> float:
        return np.std(self.returns) * np.sqrt(self._ann_factor)

    @property
    def max_drawdown(self) -> float:
        mdd = 0
        peak = self.prices.iloc[0]
        for x in self.prices:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return mdd

    @property
    def _skew(self) -> float:
        return self.returns.skew()

    @property
    def _kurtosis(self) -> float:
        return self.returns.kurtosis()

    @property
    def sharpe(self) -> float:
        return (self.ann_return - self.rfr) / self.ann_volatility

    @property
    def sortino(self) -> float:
        a_std = self.returns[self.returns < 0].std() * np.sqrt(self._ann_factor)

        return (self.ann_return - self.rfr) / a_std

    @property
    def calmar(self) -> float:
        return (self.ann_return - self.rfr) / self.max_drawdown

    @property
    def hurst(self) -> float:

        lags = range(2, 20)  # default in matlab for hurst exponent set at 20 lags

        # std-dev of differences
        tau = [np.std(np.subtract(self.prices.iloc[lag:].values, self.prices.iloc[:-lag].values)) for lag in lags]

        # slope of log plot
        regression: np.array = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst_exp = regression[:][0]

        return hurst_exp

    @handle_benchmark_comparison
    def get_beta(self) -> float:
        cmn_idx = self.returns.index.intersection(self.benchmark.pct_change().index)

        df1, df2 = self.benchmark.pct_change().loc[cmn_idx], self.returns.loc[cmn_idx]

        return np.cov(df1, df2)[0][1] / np.var(df1)

    @handle_benchmark_comparison
    def get_information_ratio(self) -> float:
        premium = self.ann_return - self.get_ann_return(self.benchmark)

        tracking_error = np.std((self.returns.values - self.benchmark.pct_change().dropna().values))

        return premium / tracking_error

    @handle_benchmark_comparison
    def get_jensens_alpha(self) -> float:
        # noinspection PyArgumentList
        return self.ann_return - self.rfr - self.get_beta() * (self.get_ann_return(self.benchmark) - self.rfr)

    def get_portfolio_stats(self, col_label: str = 'Sample Portfolio') -> pd.DataFrame:
        idx = ['return', 'volatility', 'sharpe', 'sortino', 'calmar', 'max drawdown', 'hurst-exponent',
               'skew', 'kurtosis', 'historical VaR', 'parametric VaR']

        df = pd.DataFrame(
            {col_label: [self.ann_return,
                         self.ann_volatility,
                         self.sharpe,
                         self.sortino,
                         self.calmar,
                         self.max_drawdown,
                         self.hurst,
                         self._skew,
                         self._kurtosis,
                         self.historical_var,
                         self.parametric_var
                         ]
             },
            index=idx)

        return df

    def get_performance_table(self, format_: bool = True) -> pd.DataFrame:
        year_index = sorted(
            list(map(lambda x: pd.to_datetime(x), list(set([year.strftime('%Y') for year in self.prices.index])))))

        month_index = list(
            map(lambda x: pd.to_datetime(x), monthlist_fast([self.prices.index.min(), self.prices.index.max()])))

        performance_df = pd.DataFrame(data=[],
                                      columns=['Jan',
                                               'Feb',
                                               'Mar',
                                               'Apr',
                                               'May',
                                               'Jun',
                                               'Jul',
                                               'Aug',
                                               'Sep',
                                               'Oct',
                                               'Nov',
                                               'Dec',
                                               'Year'],
                                      index=list(map(lambda x: x.strftime('%Y'), year_index))
                                      )

        month_frame = self.prices.resample('1m').first().sort_index(ascending=True).append(
            pd.Series(data=self.prices.loc[self.prices.index.max()], index=[month_index[0]])).pct_change().shift(
            -1).dropna()

        for idx in range(len(month_frame.index)):
            if month_frame.index[idx] > self.prices.index.max():
                continue
            else:
                val = month_frame.iloc[idx]

                month = month_index[idx].strftime('%b')
                year = month_index[idx].strftime('%Y')

                try:
                    performance_df.loc[year, month] = val
                except ValueError:
                    continue

        for y_idx in range(len(year_index)):
            date_mask = (self.prices.index >= year_index[y_idx]) & (
                    self.prices.index < year_index[y_idx + 1]) if y_idx != len(year_index) - 1 else (
                    self.prices.index >= year_index[y_idx])

            year_df = pd.DataFrame(self.prices.loc[date_mask].sort_index(ascending=True))

            val = (year_df.iloc[-1, 0] / year_df.iloc[0, 0]) - 1

            performance_df.loc[performance_df.index[y_idx], 'Year'] = val

        if format_:
            performance_df = performance_df.apply(
                lambda x: x.apply(lambda y: '{:.3%}'.format(y) if not np.isnan(y) else ''))
        else:
            pass

        return performance_df.sort_index(ascending=True)

    def get_month_stats(self) -> pd.DataFrame:
        month_stats_df = pd.DataFrame(
            self.get_performance_table(format_=False).iloc[:, :-1].values.reshape(-1, 1)).replace(0,
                                                                                                  np.nan).dropna().iloc[
                         :, -1]

        month_mean = month_stats_df.mean()
        month_std = month_stats_df.std()
        month_max = month_stats_df.max()
        month_min = month_stats_df.min()
        count_positive = len(month_stats_df[month_stats_df >= 0]) / len(month_stats_df)
        count_negative = 1 - count_positive

        df = pd.DataFrame(
            {'months_mean': month_mean, 'months_std': month_std, 'months_max': month_max, 'months_min': month_min,
             'count_positive': count_positive, 'count_negative': count_negative},
            index=['month_stats (%)']).T.apply(lambda x: round(x * 100, 3))

        return df

    def get_average_holding_time(self) -> dt.timedelta:
        return self.prices.loc[self.prices.diff() == 0,].drop_duplicates(
            keep='first').index.to_series().diff().mean()

    def get_percent_time_in_cash(self) -> float:
        return self.prices.loc[self.prices.diff() == 0,].shape[0] / self.prices.shape[0]
