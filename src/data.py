"""Pobieranie danych z Yahoo Finance i budowa zbioru cech."""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = {
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices",
    "KO": "Coca-Cola Company",
}


def download_prices(ticker: str, period: str = "10y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nie udało się pobrać danych dla {ticker}.")
    return df


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().dropna()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


# Etykieta anomalii (reguła k-sigma) oraz cechy z dnia t-1
def build_dataset(df: pd.DataFrame, sigma: float = 3.0):
    close = df["Close"]
    ret = close.pct_change()

    roll_mean = ret.rolling(252, min_periods=30).mean()
    roll_std = ret.rolling(252, min_periods=30).std()
    y_series = ((ret - roll_mean).abs() > sigma * roll_std).astype(int)

    vol_20 = ret.rolling(20, min_periods=10).std()
    rsi14 = _rsi(close, window=14)
    price_z = (close - close.rolling(20, min_periods=10).mean()) / (
        close.rolling(20, min_periods=10).std() + 1e-10
    )
    volume = df["Volume"].astype(float).replace(0.0, np.nan)
    vol_ratio = volume / (volume.rolling(20, min_periods=10).mean() + 1e-10)
    hl_range = (df["High"] - df["Low"]) / (df["Close"].abs() + 1e-10)

    # shift(1) - każda cecha pochodzi z dnia poprzedzającego (brak przecieku)
    features = pd.DataFrame(
        {
            "ret_lag1": ret.shift(1),
            "ret_lag2": ret.shift(2),
            "ret_lag3": ret.shift(3),
            "vol_20": vol_20.shift(1),
            "rsi14": rsi14.shift(1),
            "price_z": price_z.shift(1),
            "vol_ratio": vol_ratio.shift(1),
            "hl_range": hl_range.shift(1),
        }
    )

    valid = features.notna().all(axis=1) & y_series.notna()
    features = features[valid]
    y_clean = y_series[valid]

    X = features.values.astype(float)
    y_out = y_clean.values.astype(int)

    idx = features.index
    try:
        idx = idx.tz_convert(None)
    except TypeError:
        pass
    dates = idx.to_numpy()

    prices = close[valid].values.astype(float)
    feat_names = list(features.columns)
    return X, y_out, dates, prices, feat_names


def class_summary(y):
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    pct = 100.0 * n_pos / len(y) if len(y) > 0 else 0.0
    return n_neg, n_pos, pct
