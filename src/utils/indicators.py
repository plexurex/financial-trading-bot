import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def fetch_stock_with_indicators(symbol, period='60d', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)

    df.dropna(inplace=True)

    # Explicitly ensure "Close" is a 1-dimensional Series
    close_prices = df['Close'].squeeze()

    # RSI Calculation
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD Calculation
    macd_indicator = MACD(close=close_prices)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()

    # Bollinger Bands Calculation
    bb_indicator = BollingerBands(close=close_prices)
    df['Bollinger_high'] = bb_indicator.bollinger_hband()
    df['Bollinger_low'] = bb_indicator.bollinger_lband()

    # Simple Moving Average (SMA)
    sma_indicator = SMAIndicator(close=close_prices, window=20)
    df['SMA_20'] = sma_indicator.sma_indicator()

    df.dropna(inplace=True)

    return df
