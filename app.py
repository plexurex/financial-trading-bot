import streamlit as st
from src.utils.data_collection import fetch_stock_data
from src.utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from src.utils.trading_strategy import decide_trade

st.title("Financial Trading Bot")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")
if st.button("Get Recommendation"):
    with st.spinner("Fetching data and analyzing..."):
        stock_data = fetch_stock_data(symbol)
        news_sentiment = analyze_news_sentiment(symbol)
        social_sentiment = analyze_social_sentiment(symbol)

        action = decide_trade(news_sentiment, social_sentiment)
        st.write(f"Decision for {symbol}: {action}")

        st.subheader("Detailed Results")
        st.write(f"News Sentiment: {news_sentiment}")
        st.write(f"Social Media Sentiment: {social_sentiment}")

        st.subheader("Recent Stock Data")
        st.dataframe(stock_data.tail())
