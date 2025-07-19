# By Advance  in  Financial  Machine  Learning, Machine Learning for Asset Managers
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
import statsmodels.api as sm1


# HHI
def getHHI(betRet):
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght**2).sum1()
    hhi = (hhi - betRet.shape[0] ** -1) / (1.0 - betRet.shape[0] ** -1)
    return hhi


def getHHI_PN(betRet):  # 正負分離、高いとファットテール傾向
    return {
        "HHI_Positive": getHHI(betRet[betRet >= 0]),
        "HHI_Negative": getHHI(betRet[betRet < 0]),
    }


# DeNoise of corr
def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def getPCA(matrix):
    # エルミート行列からeVal, eVecを取得
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # 降順
    eVal = eVal[indices]
    eVec = eVec[indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec


# トレンドスキャン
def tValLinR(close):
    # 線形トレンドからのt値
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm1.OLS(close, x).fit()
    return ols.tvalues[1]


def getBinsFromTrend(molecule, close, span=[3, 10, 1]):
    """
    線形トレンドのt値の符号からラベルを作成
    - t1: 発見されたトレンドの終了時点
    - tVal: 推定トレンド回帰のt値
    - bin: トレンドの符号
    """
    out = pd.DataFrame(index=molecule, columns=["t1", "tVal", "bin"])
    hrzns = range(*span)
    for dt0 in molecule:
        df0 = pd.Series(dtype="float64")
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(hrzns) > close.shape[0]:
            continue
        for hrzn in hrzns:
            dt1 = close.index[iloc0 + hrzn - 1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = tValLinR(df1.values)
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ["t1", "tVal", "bin"]] = (
            df0.index[-1],
            df0[dt1],
            np.sign(df0[dt1]),
        )  # リーケージを回避
    out["t1"] = pd.to_datetime(out["t1"])
    out["bin"] = pd.to_numeric(out["bin"], downcast="signed")
    return out.dropna(subset=["bin"])


def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1


# Mutual Information
def numBins(nObs, corr=None):
    # 離散化の最適ビン数
    if corr == None:  # 単変量の場合
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs**2) ** 0.5) ** (
            1 / 3.0
        )  # Hacine-Gharbi
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:  # ２変量の場合
        b = round(2**-0.5 * (1 + (1 + 24 * nObs / (1.0 - corr**2)) ** 0.5) ** 0.5)
    return int(b)


def varInfo(x, y, norm=False):
    # VI
    bXY = numBins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = ss.entropy(np.histogram(x, bXY)[0])  # 周辺エントロピー
    hY = ss.entropy(np.histogram(y, bXY)[0])  # 周辺エントロピー
    vXY = hX + hY - 2 * iXY  # VI
    if norm:
        hXY = hX + hY - iXY  # 結合エントロピー
        vXY /= hXY
    return vXY


def mutualInfo(x, y, norm=False):
    # 相互情報量
    bXY = numBins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    if norm:
        hX = ss.entropy(np.histogram(x, bXY)[0])  # 周辺エントロピー
        hY = ss.entropy(np.histogram(y, bXY)[0])  # 周辺エントロピー
        iXY /= min(hX, hY)  # 正規化相互情報量
    return iXY
