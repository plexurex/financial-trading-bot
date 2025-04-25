import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def fetch_stock_with_indicators(symbol, period='60d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    
    # Print debug information
    print(f"Downloaded {len(df)} rows for {symbol}")
    print(f"Original columns: {df.columns}")
    
    # Check if data was fetched
    if df.empty:
        print(f"No data found for {symbol}")
        return df
    
    # Handle multi-level columns that sometimes come from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        print(f"Converting MultiIndex columns for {symbol}")
        df.columns = [f"{col[0]}_{symbol}" if col[1] == symbol else f"{col[0]}" for col in df.columns]
        print(f"After conversion columns: {df.columns}")
        
        # Make sure we have a 'Close' column
        if f'Close_{symbol}' in df.columns and 'Close' not in df.columns:
            df['Close'] = df[f'Close_{symbol}']
            print(f"Using Close_{symbol} as Close column")

    df.dropna(inplace=True)

    # Explicitly ensure "Close" is a 1-dimensional Series
    close_prices = df['Close'].squeeze()

    # RSI Calculation
    print(f"Adding RSI for {symbol}...")
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD Calculation
    print(f"Adding MACD for {symbol}...")
    macd_indicator = MACD(close=close_prices)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()

    # Bollinger Bands Calculation
    print(f"Adding Bollinger Bands for {symbol}...")
    bb_indicator = BollingerBands(close=close_prices)
    df['Bollinger_high'] = bb_indicator.bollinger_hband()
    df['Bollinger_low'] = bb_indicator.bollinger_lband()

    # Simple Moving Average (SMA)
    print(f"Adding SMA for {symbol}...")
    sma_indicator20 = SMAIndicator(close=close_prices, window=20)
    df['SMA_20'] = sma_indicator20.sma_indicator()
    
    # Add SMA 50 and SMA 10 for additional trend information
    sma_indicator50 = SMAIndicator(close=close_prices, window=50)
    df['SMA_50'] = sma_indicator50.sma_indicator()
    
    sma_indicator10 = SMAIndicator(close=close_prices, window=10)
    df['SMA_10'] = sma_indicator10.sma_indicator()

    # Remove rows with NaN values after adding indicators
    nan_count_before = df.isna().sum().sum()
    df.dropna(inplace=True)
    nan_count_after = len(df)
    print(f"Removed {nan_count_before - nan_count_after} rows with NaN values for {symbol}")
    print(f"Final columns for {symbol}: {df.columns.tolist()}")

    return df