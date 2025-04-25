# src/models/train_multi_horizon.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.utils.data_collection import fetch_stock_data

def prepare_targets(df, horizons):
    """
    For each h in horizons, create Target_h = 1 if Close_h > Close_0 else 0.
    """
    for h in horizons:
        df[f"Target_{h}"] = (df["Close"].shift(-h) - df["Close"] > 0).astype(int)
    return df.dropna()

def main():
    # Ten major non-meme cryptos
    cryptos = [
        "BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD",
        "SOL-USD","DOT-USD","MATIC-USD","AVAX-USD","LTC-USD"
    ]
    horizons = [1, 30, 60, 252]
    features = ["RSI","MACD","MACD_signal","Bollinger_high","Bollinger_low","SMA_20"]

    # Make sure data folder exists
    base_dir = os.path.dirname(__file__)        # src/models
    data_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    for symbol in cryptos:
        print(f"\n=== Training {symbol} ===")
        df = fetch_stock_data(symbol, period="5y", interval="1d")
        if df.empty:
            print(f" ⚠️ no data for {symbol}, skipping")
            continue

        df = prepare_targets(df, horizons)
        df = df[features + [f"Target_{h}" for h in horizons]]

        sym_safe = symbol.replace("-", "_")  # BTC-USD → BTC_USD

        for h in horizons:
            X = df[features]
            y = df[f"Target_{h}"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            print(f"  Horizon {h}d accuracy: {acc:.2%}")

            out_path = os.path.join(data_dir, f"{sym_safe}_h{h}.joblib")
            joblib.dump(clf, out_path)
            print(f"   ⇒ saved model at {out_path}")

if __name__ == "__main__":
    main()
