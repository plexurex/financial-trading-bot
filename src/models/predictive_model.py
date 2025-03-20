import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

class TradingPredictor:
    def __init__(self, model_path='trading_model.joblib'):
        self.model_path = model_path
        try:
            self.model = load(model_path)
        except:
            self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)
        dump(self.model, self.model_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
