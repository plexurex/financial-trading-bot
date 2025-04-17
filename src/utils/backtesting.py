import pandas as pd
import numpy as np

def backtest_strategy(df):
    """
    This function backtests a simple strategy based on the 20-day Simple Moving Average.
    A signal is generated: 1 if Close > SMA_20 (in the market), 0 otherwise.
    The strategy return is computed using yesterday's signal multiplied by today's daily return.
    It then calculates cumulative returns, Sharpe ratios, and maximum drawdown.
    """
    # Create a copy so we don't modify the original DataFrame
    df = df.copy()

    # Ensure the 'Close' and 'SMA_20' columns are one-dimensional Series
    close_series = df['Close'].squeeze()
    sma_series = df['SMA_20'].squeeze()
    
    # Align the two Series by their index
    aligned_close, aligned_sma = close_series.align(sma_series, join='inner', copy=False)
    
    # Restrict the DataFrame to the aligned index
    df_aligned = df.loc[aligned_close.index].copy()
    
    # Generate the signal: 1 if Close > SMA_20, else 0
    df_aligned['signal'] = (aligned_close > aligned_sma).astype(int)

    # Compute daily return as the percentage change in 'Close' prices
    df_aligned['daily_return'] = df_aligned['Close'].pct_change()

    # Compute strategy return: apply yesterday's signal to today's return
    df_aligned['strategy_return'] = df_aligned['signal'].shift(1) * df_aligned['daily_return']

    # Calculate cumulative returns: for both the strategy and a buy-and-hold approach
    df_aligned['cumulative_strategy'] = (1 + df_aligned['strategy_return']).cumprod()
    df_aligned['cumulative_buyhold'] = (1 + df_aligned['daily_return']).cumprod()

    # Calculate the annualized Sharpe Ratio (assume 252 trading days, risk-free rate=0)
    if df_aligned['strategy_return'].std() != 0:
        sharpe_strategy = (df_aligned['strategy_return'].mean() / df_aligned['strategy_return'].std()) * np.sqrt(252)
    else:
        sharpe_strategy = np.nan

    if df_aligned['daily_return'].std() != 0:
        sharpe_buyhold = (df_aligned['daily_return'].mean() / df_aligned['daily_return'].std()) * np.sqrt(252)
    else:
        sharpe_buyhold = np.nan

    # Calculate Maximum Drawdown: the minimum difference between cumulative return and its running maximum
    cummax = df_aligned['cumulative_strategy'].cummax()
    df_aligned['drawdown'] = (df_aligned['cumulative_strategy'] - cummax) / cummax
    max_drawdown = df_aligned['drawdown'].min()

    # Pack performance metrics into a dictionary
    metrics = {
        'cumulative_strategy_return': df_aligned['cumulative_strategy'].iloc[-1] - 1,
        'cumulative_buyhold_return': df_aligned['cumulative_buyhold'].iloc[-1] - 1,
        'sharpe_ratio_strategy': sharpe_strategy,
        'sharpe_ratio_buyhold': sharpe_buyhold,
        'max_drawdown_strategy': max_drawdown
    }
    
    return df_aligned, metrics
