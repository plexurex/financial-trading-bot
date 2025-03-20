import pandas as pd
from sklearn.model_selection import train_test_split
from .predictive_model import TradingPredictor
from ..utils.data_collection import fetch_stock_data

def prepare_data(df):
    df['Price_Direction'] = df['Close'].diff().shift(-1)
    df.dropna(inplace=True)
    df['Target'] = (df['Price_Direction'] > 0).astype(int)
    features = ['RSI', 'MACD', 'MACD_signal', 'Bollinger_high', 'Bollinger_low', 'SMA_20']
    return df[features], df['Target']

def main():
    symbol = "AAPL"
    df = fetch_stock_data(symbol)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    predictor = TradingPredictor()
    predictor.train(X_train, y_train)

    accuracy = predictor.model.score(X_test, y_test)
    print(f"Model trained successfully. Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
