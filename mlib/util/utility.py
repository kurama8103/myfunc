__version__ = "0.0.1"

import csv
import io
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from numpy import set_printoptions
import matplotlib.pyplot as plt
from seaborn import set_style
import japanize_matplotlib
import requests


def pref():
    # pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100
    # pd.options.display.width = 120
    # pd.options.display.float_format = "{:,.4f}".format

    set_printoptions(suppress=True)
    # set_printoptions(precision=2)

    plt.figure()
    plt.rcParams["figure.figsize"] = 12, 4
    # rcParams['font.family']= 'Yu Mincho'

    set_style("whitegrid")
    japanize_matplotlib.japanize()


def end_of_month(df):
    if type(df) is not pd.DatetimeIndex:
        df = df.index
    df = df.sort_values()
    return df[(pd.Series(df.month.values).diff(-1) != 0).values]


def apply_concat(df, func, axis=0):
    return pd.concat(
        [func(df[x].dropna()) for x in df.columns.values],
        axis=axis,
        keys=df.columns.values,
    )


def compare_by_func(x, y=None, ax=None, figsize=(6, 6), correlation="pearson"):
    if y is None:
        if ax is None:
            plt.hist(x, bins=20)
        else:
            ax.hist(x, bins=20)
    else:
        from yellowbrick.features import joint_plot

        joint_plot(x, y, ax=ax, figsize=figsize, show=False, correlation=correlation)


def load_test_data():
    from statsmodels.datasets import macrodata

    df = macrodata.load()["data"]
    df.index = pd.date_range("1959-03-31", periods=len(df), freq="Q")
    df[df.columns.drop(["year", "quarter"])] = (
        df[df.columns.drop(["year", "quarter"])].ffill().pct_change()
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.drop("realgdp", axis=1), df["realgdp"]


import os


def load_test_data_fred():
    print(__file__)


from numpy.lib.stride_tricks import sliding_window_view


def apply_moving_window(x, func, window: int = 10) -> list:
    return [func(d) for d in sliding_window_view(x, window, axis=0)]


def apply_moving_window_df(x: pd.DataFrame, func, window: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        apply_moving_window(x.values, func=func, window=window),
        index=x.tail(len(x) + 1 - window).index,
    )


def vis_func(func, range=10, n=100, plot=True):
    step = range * 2 / n
    x = np.arange(-range, range, step=step).round(8)
    try:
        y = np.hstack(func(x))
    except:
        y = [func(i) for i in x]
    if plot:
        plt.plot(x, y)
    return x, y


from mlib.finance.stochastic import ornstein_uhlenbeck_process


def vis_func_array(func, n=100, plot=True):
    window = (n * 5) // 10

    x_brownian = np.random.randn(n) / 100
    x_trend = np.arange(n) / n + x_brownian
    # x_anti_trend = np.exp(-x_trend) + x_brownian
    _ = ornstein_uhlenbeck_process(100, 100, T=5, M=100, npath=1)
    x_anti_trend = pd.DataFrame(_).pct_change().dropna().values.reshape(-1)
    _ = pd.DataFrame(
        {
            "f_trend": apply_moving_window(x_trend, func, window),
            "f_brownian": apply_moving_window(x_brownian, func, window),
            "f_anti_trend": apply_moving_window(x_anti_trend, func, window),
        }
    )
    if plot:
        _.plot()
    return _


def autocorr(x):
    corr = np.correlate(x, x, mode="full")
    return corr[int(corr.size / 2) :]


def class_to_csv(class_):
    f = io.StringIO()
    writer = csv.DictWriter(f, fieldnames=class_.__dict__.keys())
    writer.writeheader()
    writer.writerow(class_.__dict__)
    return f


def get_dtypes(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}


def reduce_float(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    c = df.select_dtypes(include=["float64", "float32"]).columns
    df[c] = df[c].astype("float16")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def send_to_discord(webhook_url, message):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("send: " & message)
    else:
        print(f"{response.status_code}, {response.text}")
