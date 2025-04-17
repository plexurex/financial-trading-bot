import yfinance as yf
import pandas as pd
import numpy as np

def fetch_multi_asset_data(tickers, period='1y', interval='1d'):
    """
    Fetch historical data for a list of tickers using yfinance.
    Returns a dictionary mapping each ticker to its DataFrame.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            data[ticker] = df
    return data

def compute_signals(df, window=20):
    """
    Compute a simple moving average (SMA)-based signal for an individual asset.
    Signal = 1 if the asset's Close price is above its SMA, else 0.
    
    This function computes the SMA, then forces the 'Close' and 'SMA' columns
    to be one-dimensional Series and aligns them along the index (axis=0) before 
    computing the signal.
    """
    df = df.copy()
    # Compute the 20-day SMA and store in column 'SMA'
    df['SMA'] = df['Close'].rolling(window=window).mean()
    
    # Force 'Close' and 'SMA' to be Series
    close = df['Close'].squeeze()
    sma = df['SMA'].squeeze()
    
    # Align the two series on their index (inner join)
    aligned_close, aligned_sma = close.align(sma, join='inner', copy=False, axis=0)
    
    # Compute the trading signal: 1 if aligned_close > aligned_sma, else 0
    signal = (aligned_close > aligned_sma).astype(int)
    
    # Restrict df to the aligned indices and add the signal column
    df_aligned = df.loc[aligned_close.index].copy()
    df_aligned['signal'] = signal

    # Drop any rows with NaNs resulting from the rolling calculation
    return df_aligned.dropna()

def simulate_portfolio(data_dict, initial_capital=10000, window=20):
    """
    Simulate a multi-asset adaptive trading strategy that holds positions over multiple days.
    
    For each asset:
      - The initial capital is allocated equally.
      - If an asset’s signal is 1 (BUY) and you are not invested (shares == 0),
        buy at the close price and hold until the signal changes.
      - If the signal is 0 (SELL) while holding shares, sell to convert back to cash.
    
    The function calculates the portfolio value over time for each asset and then
    combines them over the common dates.
    
    Returns:
      - combined_portfolio: a DataFrame with the overall portfolio value indexed by date.
    """
    # Compute signals for each asset
    dfs = {}
    for ticker, df in data_dict.items():
        dfs[ticker] = compute_signals(df, window=window).sort_index()

    # Dictionary to hold portfolio value series for each asset
    portfolio_values = {}
    for ticker, df in dfs.items():
        # Initialize cash and holdings for this asset
        cash = initial_capital / len(data_dict)
        shares = 0.0
        # Prepare a Series to track portfolio value over time
        pv = pd.Series(index=df.index, dtype=float)
        
        for date in df.index:
            # Get today's closing price and ensure it's a scalar
            price = df.loc[date, 'Close']
            if not np.isscalar(price):
                price = price.item() if hasattr(price, 'item') else price.iloc[0]
            
            # Get the signal and make sure it's a scalar
            signal = df.loc[date, 'signal']
            if not np.isscalar(signal):
                signal = signal.item() if hasattr(signal, 'item') else signal.iloc[0]
            
            # Trading logic:
            # - If signal is 1 and no shares are held, buy at the closing price.
            if signal == 1 and shares == 0:
                shares = cash / price
                cash = 0.0
            # - If signal is 0 and shares are held, sell all to convert to cash.
            elif signal == 0 and shares > 0:
                cash = shares * price
                shares = 0.0

            # Update the portfolio value (cash + market value of shares)
            pv.loc[date] = cash + shares * price
        
        portfolio_values[ticker] = pv

    # Find common dates across all asset portfolio series
    common_dates = set.intersection(*(set(pv.index) for pv in portfolio_values.values()))
    common_dates = sorted(common_dates)
    
    # Combine the portfolio values from each asset into an overall portfolio series
    combined_portfolio = pd.DataFrame(index=common_dates, columns=['portfolio_value'])
    for date in common_dates:
        total_val = sum(portfolio_values[ticker].loc[date] for ticker in portfolio_values if date in portfolio_values[ticker].index)
        combined_portfolio.loc[date, 'portfolio_value'] = total_val

    return combined_portfolio

