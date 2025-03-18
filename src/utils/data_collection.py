import yfinance as yf

def fetch_stock_data(symbol, period="1mo", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    return data
