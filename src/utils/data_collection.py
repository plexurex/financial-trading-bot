from .indicators import fetch_stock_with_indicators

def fetch_stock_data(symbol, period='60d', interval='1d'):
    return fetch_stock_with_indicators(symbol, period, interval)
