import streamlit as st
from src.utils.data_collection import fetch_stock_data
from src.utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from src.utils.trading_strategy import decide_trade, risk_managed_decision
from src.models.predictive_model import TradingPredictor
from src.models.rl_agent import load_rl_agent

st.title("🚀 Advanced Financial Trading Bot")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")
if st.button("Analyze and Trade"):
    with st.spinner("Fetching data and analyzing..."):
        df = fetch_stock_data(symbol, period='60d')
        latest_data = df.iloc[-1:]
        current_price = latest_data['Close'].values[0]

        news_sentiment = analyze_news_sentiment(symbol)
        social_sentiment = analyze_social_sentiment(symbol)

        predictor = TradingPredictor()
        features = ['RSI', 'MACD', 'MACD_signal', 'Bollinger_high', 'Bollinger_low', 'SMA_20']
        ml_pred = predictor.predict(latest_data[features])[0]
        proba = predictor.predict_proba(latest_data[features])[0][ml_pred]

        ml_decision = "BUY" if ml_pred == 1 else "SELL"
        sentiment_decision = decide_trade(news_sentiment, social_sentiment)

        # Load RL model
        rl_model, rl_env = load_rl_agent(df)
        obs = rl_env.reset()
        rl_action, _ = rl_model.predict(obs)
        rl_decision = ["HOLD", "BUY", "SELL"][rl_action]

        # Final combined decision
        combined_decision = ml_decision if ml_decision == sentiment_decision == rl_decision else "HOLD"

        # Risk-managed decision (assuming entry price from 5 days ago)
        entry_price = df['Close'].iloc[-5]
        final_decision = risk_managed_decision(combined_decision, current_price, entry_price)

        st.header("📈 Trading Decision")
        st.write(f"**Final Risk-Managed Decision:** {final_decision}")

        st.subheader("🔍 Decision Breakdown")
        st.write(f"- **ML Prediction:** {ml_decision} ({proba:.2%} confidence)")
        st.write(f"- **Sentiment Decision:** {sentiment_decision}")
        st.write(f"- **RL Agent Decision:** {rl_decision}")

        # Display the raw individual sentiments for clarity
        st.subheader("📝 Raw Sentiment Values")
        st.write(f"- **News Sentiment:** {news_sentiment:.4f}")
        st.write(f"- **Reddit Sentiment:** {social_sentiment:.4f}")

        st.subheader("📊 Latest Market Data")
        st.dataframe(df.tail())
