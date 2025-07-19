# %%
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import norm
from numpy.linalg import inv


def norm_integral(f, mean, std):
    val, er = quad(
        lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
        mean - 3 * std,
        mean + 3 * std,
    )
    return -val


def get_kelly_share(data, bounds=[0, 1]):
    solution = minimize_scalar(
        norm_integral,
        args=(data["mean"], data["std"]),
        bounds=bounds,
        method="bounded",
    )
    return solution.x


def vis_kelly_return(y):
    return_params = y.rolling(4).agg(["mean", "std"]).dropna()
    k = return_params.apply(get_kelly_share, axis=1)
    # k = get_kelly_share(return_params["mean"], return_params["std"])
    l = np.sign(return_params["mean"])
    df = pd.DataFrame(y).assign(
        wgt_kelly=k.shift(),
        rt_kelly=y.mul(k.shift()),
        wgt_sign=l.shift(),
        rt_sign=y.mul(l.shift()),
    )
    (df.drop(["wgt_kelly", "wgt_sign"], axis=1).dropna() + 1).cumprod().plot()
    df[["wgt_kelly", "wgt_sign"]].plot()
    return df


def vis_kelly_mesh(bounds=[-2, 3]):
    f = np.vectorize(
        lambda i, j: minimize_scalar(
            norm_integral, args=(i, j), bounds=bounds, method="bounded"
        ).x
    )
    i, j = np.meshgrid(np.linspace(-0.1, 0.1, 10), np.linspace(0.01, 0.2, 20))
    k = f(i, j)
    sns.heatmap(
        pd.DataFrame(k, columns=i[0].round(2), index=j[:, 0].round(2)),
        annot=True,
        center=0,
    )
    return k


def wgt_kelly(mean, covariance):
    wgt_kelly = inv(covariance).dot(mean)
    wgt_kelly /= sum(wgt_kelly)
    return wgt_kelly


def kelly_criterion_1d(probability, e_return, limit: list = None):
    """
    :param p: the probability of a win.
    :param b: the proportion of the bet gained with a win.
    :return: the fraction of the current bankroll to wager.
    """
    weight = np.where(e_return > 0, (probability * (e_return + 1) - 1) / e_return, 0)
    if limit:
        weight = np.where(weight > limit[1], limit[1], weight)
        weight = np.where(weight < limit[0], limit[0], weight)
    return weight


# %%
if __name__ == "__main__":
    import mlib

    X, y = mlib.load_test_data()
    # mlib.vis_func(lambda x: norm_integral(x, 0.05, 0.2))
    vis_kelly_mesh(bounds=[-1, 1])
    vis_kelly_return(y - 0.01)

    dfr = X.iloc[:, 2:]
    mu = dfr.mean() * 4
    covar = dfr.cov() * 4
    wgt = wgt_kelly(mu, covar)
    print(pd.Series(wgt, dfr.columns), wgt.dot(mu), np.sqrt(wgt.dot(covar).dot(wgt)))

    # mlib.vis_func(lambda x: kelly_criterion_1d(0.5, x), range=2)
    # mlib.vis_func(lambda x: kelly_criterion_1d(0.55, x), range=2)
# %%
