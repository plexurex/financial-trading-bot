# src/utils/data_collection.py

import pandas as pd
import yfinance as yf
from functools import lru_cache

@lru_cache(maxsize=16)
def fetch_stock_data(symbol: str,
                     period: str = "1y",
                     interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for `symbol` from Yahoo via yfinance.
    Caches the last 16 calls in memory to speed up repeated calls
    in the same Python process.
    """
    # yfinance wants no dash for crypto, but it actually handles "BTC-USD" fine.
    # If you run into issues you can strip "-USD" off for cryptos:
    # sym = symbol.replace("-USD", "")
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
    # ensure the index is a DatetimeIndex named like your downstream code expects
    df = df.loc[:, ["Open","High","Low","Close","Volume"]]
    df.index.name = None
    return df
