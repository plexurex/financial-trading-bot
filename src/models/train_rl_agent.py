from ..utils.data_collection import fetch_stock_data
from .rl_agent import train_rl_agent

def main():
    symbol = "AAPL"
    df = fetch_stock_data(symbol, period='1y')
    train_rl_agent(df)

if __name__ == "__main__":
    main()