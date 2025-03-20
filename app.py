import streamlit as st
from src.utils.data_collection import fetch_stock_data
from src.utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from src.utils.trading_strategy import decide_trade
from src.models.predictive_model import TradingPredictor

st.title("🚀 Financial Trading Bot")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")
if st.button("Get Recommendation"):
    with st.spinner("🔄 Fetching data and analyzing..."):
        # Fetch latest stock data with indicators
        df = fetch_stock_data(symbol)
        latest_data = df.iloc[-1:]

        # Sentiment analysis
        news_sentiment = analyze_news_sentiment(symbol)
        social_sentiment = analyze_social_sentiment(symbol)

        # Load trained ML model
        predictor = TradingPredictor()
        features = ['RSI', 'MACD', 'MACD_signal', 'Bollinger_high', 'Bollinger_low', 'SMA_20']
        prediction = predictor.predict(latest_data[features])[0]
        proba = predictor.predict_proba(latest_data[features])[0][prediction]

        ml_decision = "BUY" if prediction == 1 else "SELL"
        sentiment_decision = decide_trade(news_sentiment, social_sentiment)
        
        # Combine decisions clearly
        final_decision = ml_decision if ml_decision == sentiment_decision else "HOLD"

        # Display clearly
        st.write(f"## 📈 Final Decision: **{final_decision}**")

        st.write("### 🧠 ML Model Prediction:")
        st.write(f"- **Decision:** {ml_decision}")
        st.write(f"- **Confidence:** {proba:.2%}")

        st.write("### 📰 Sentiment Analysis:")
        st.write(f"- **News sentiment:** {news_sentiment:.4f}")
        st.write(f"- **Social media sentiment:** {social_sentiment:.4f}")

        st.write("### 📊 Recent Stock Data:")
        st.dataframe(df.tail())
