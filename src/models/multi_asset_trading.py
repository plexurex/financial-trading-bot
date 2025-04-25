import yfinance as yf
import pandas as pd
import numpy as np

def fetch_multi_asset_data(tickers, period='1y', interval='1d'):
    """
    Fetch historical data for a list of tickers using yfinance.
    Returns a dict mapping each ticker to its DataFrame.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            data[ticker] = df
    return data

def compute_signals(df, window=20):
    """
    Compute a simple moving average (SMA)-based signal for an asset.
    Signal = 1 if Close > SMA, else 0.
    Aligns the Close and SMA series to avoid misalignment.
    """
    df = df.copy()
    # 1) Compute the SMA
    df['SMA'] = df['Close'].rolling(window=window).mean()

    # 2) Force them into 1D Series and align by index
    close = df['Close'].squeeze()
    sma   = df['SMA'].squeeze()
    aligned_close, aligned_sma = close.align(sma, join='inner', copy=False, axis=0)

    # 3) Build an aligned DataFrame and generate the signal
    df_aligned = df.loc[aligned_close.index].copy()
    df_aligned['signal'] = (aligned_close > aligned_sma).astype(int)

    # 4) Drop any NaNs introduced by the rolling mean
    return df_aligned.dropna()

def simulate_portfolio(data_dict, initial_capital=10000, window=20):
    """
    Simulate a multi-asset strategy that holds positions over days.
    
    - Capital is split equally across assets.
    - If signal=1 and no shares held, buy at that day's close.
    - If signal=0 and shares held, sell at that day's close.
    - Portfolio value each day = cash + shares * close.
    - Returns a DataFrame of combined portfolio value over common dates.
    """
    num_assets = len(data_dict)
    allocation  = initial_capital / num_assets
    portfolio_values = {}

    # 1) Simulate each asset individually
    for ticker, df in data_dict.items():
        df_sig = compute_signals(df, window=window).sort_index()
        cash, shares = allocation, 0.0
        pv = pd.Series(index=df_sig.index, dtype=float)

        for date in df_sig.index:
            # Pull the full row (may be Series or tiny DataFrame)
            row = df_sig.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            price  = float(row['Close'])
            signal = int(row['signal'])

            # Buy if signal=1 and not already invested
            if signal == 1 and shares == 0:
                shares = cash / price
                cash = 0.0
            # Sell if signal=0 and shares held
            elif signal == 0 and shares > 0:
                cash   = shares * price
                shares = 0.0

            # Record portfolio value for this asset
            pv.loc[date] = cash + shares * price

        portfolio_values[ticker] = pv

    # 2) Combine across assets on common dates
    common_dates = set.intersection(*(set(s.index) for s in portfolio_values.values()))
    common_dates = sorted(common_dates)
    combined = pd.DataFrame(index=common_dates, columns=['portfolio_value'], dtype=float)

    for date in common_dates:
        total = sum(portfolio_values[t].loc[date] for t in portfolio_values)
        combined.loc[date, 'portfolio_value'] = total

    return combined
