import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.utils.data_collection import fetch_stock_data

# TA imports
from ta.momentum    import RSIIndicator
from ta.trend       import MACD, EMAIndicator
from ta.volatility  import BollingerBands, AverageTrueRange


def standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename any MultiIndex-flattened 'Close_BTC-USD', 'High_BTC-USD', etc.
    back to the five standard price columns: Open, High, Low, Close, Volume.
    """
    price_keys = ["Open", "High", "Low", "Close", "Volume"]
    rename_map = {}
    for col in df.columns:
        for pk in price_keys:
            if col.startswith(f"{pk}_"):
                rename_map[col] = pk
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a suite of technical indicators:
      - RSI (14)
      - MACD (12,26,9)
      - Bollinger Bands (20,2)
      - ATR (14)
      - EMA (20)
    Drops any rows that become NaN after indicator calculation.
    """
    # RSI
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    # MACD
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    # Bollinger Bands
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["Bollinger_high"] = bb.bollinger_hband()
    df["Bollinger_low"]  = bb.bollinger_lband()
    # ATR
    df["ATR_14"] = AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()
    # EMA
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()

    
    return df.dropna()

def prepare_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    For each horizon h, create binary target: 1 if Close shifted -h > Close now.
    """
    for h in horizons:
        df[f"Target_{h}"] = (
            df["Close"].shift(-h) - df["Close"] > 0
        ).astype(int)
    
    return df.dropna()

def main():
    # 10 large non-meme cryptos
    cryptos = [
        "BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD",
        "SOL-USD","DOT-USD","MATIC-USD","AVAX-USD","LTC-USD"
    ]
    horizons = [1, 30, 60, 252]
    features = [
        "RSI","MACD","MACD_signal",
        "Bollinger_high","Bollinger_low",
        "ATR_14","EMA_20"
    ]

    # data directory (../data)
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    for symbol in cryptos:
        print(f"\n=== Training {symbol} ===")
        raw = fetch_stock_data(symbol, period="5y", interval="1d")
        if raw.empty:
            print(f" ⚠️ no data for {symbol}, skipping")
            continue

        # flatten MultiIndex price columns and standardize names
        df = standardize_price_columns(raw)

        # add technical indicators
        try:
            df = add_technical_features(df)
        except Exception as e:
            print(f" ✖ technical feature error for {symbol}: {e}")
            continue

        # build targets
        df = prepare_targets(df, horizons)

        # keep only the features + target columns
        df = df[features + [f"Target_{h}" for h in horizons]]

        # train one model per horizon
        sym_safe = symbol.replace("-", "_")  # e.g. BTC_USD
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
