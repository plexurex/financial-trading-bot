from utils.data_collection import fetch_stock_data
from utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from utils.trading_strategy import decide_trade
from models.predictive_model import TradingPredictor

def main(symbol):
    df = fetch_stock_data(symbol)
    latest_data = df.iloc[-1:]

    news_sentiment = analyze_news_sentiment(symbol)
    social_sentiment = analyze_social_sentiment(symbol)

    predictor = TradingPredictor()
    features = ['RSI', 'MACD', 'MACD_signal', 'Bollinger_high', 'Bollinger_low', 'SMA_20']
    prediction = predictor.predict(latest_data[features])[0]
    proba = predictor.predict_proba(latest_data[features])[0][prediction]

    ml_decision = "BUY" if prediction == 1 else "SELL"

    sentiment_decision = decide_trade(news_sentiment, social_sentiment)

    print(f"Stock: {symbol}")
    print(f"ML Model Prediction: {ml_decision} ({proba:.2%} confidence)")
    print(f"Sentiment Decision: {sentiment_decision}")

    final_decision = ml_decision if ml_decision == sentiment_decision else "HOLD"
    print(f"Final Decision: {final_decision}")

if __name__ == "__main__":
    main("AAPL")
