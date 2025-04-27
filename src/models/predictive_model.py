# src/models/predictive_model.py

import os
import joblib

class MultiHorizonPredictor:
    """
    Loads 4 RandomForest models for a given crypto symbol,
    expecting them in src/data/ with names like BTC_USD_h1.joblib.
    """

    def __init__(self, symbol, data_dir=None):
        # default data_dir → ../data relative to this file
        if data_dir is None:
            base = os.path.dirname(__file__)         # src/models
            data_dir = os.path.normpath(os.path.join(base, "..", "data"))

        # convert dash→underscore to match filenames
        self.symbol = symbol.replace("-", "_")       # e.g. "BTC-USD" → "BTC_USD"
        self.horizons = [1, 30, 60, 252]
        self.models = {}

        for h in self.horizons:
            fname = f"{self.symbol}_h{h}.joblib"
            path  = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing model for {symbol} horizon={h}d at {path}"
                )
            self.models[h] = joblib.load(path)

    def predict(self, X, horizon):
        """
        X: pd.DataFrame with a single row of features.
        horizon: 1, 30, 60, or 252
        returns (pred_class:int, pred_proba:float)
        """
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon} not supported")
        clf = self.models[horizon]
        pred = int(clf.predict(X)[0])
        proba_up = float(clf.predict_proba(X)[0][1])
        return pred, proba_up
