import streamlit as st
import matplotlib.pyplot as plt

from src.utils.data_collection import fetch_stock_data
from src.utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from src.utils.trading_strategy import decide_trade, risk_managed_decision
from src.models.predictive_model import TradingPredictor
from src.models.rl_agent import load_rl_agent
from src.utils.backtesting import backtest_strategy
from src.models.multi_asset_trading import fetch_multi_asset_data, simulate_portfolio
st.title("🚀 Advanced Financial Trading Bot")

# Real-time Trading Analysis Section
symbol = st.text_input("Enter Stock Symbol:", "AAPL")

if st.button("Analyze and Trade"):
    with st.spinner("Fetching data and analyzing..."):
        # Fetch the latest 60 days of data
        df = fetch_stock_data(symbol, period='60d')
        latest_data = df.iloc[-1:]
        current_price = latest_data['Close'].values[0]
        
        # Sentiment Analysis
        news_sentiment = analyze_news_sentiment(symbol)
        social_sentiment = analyze_social_sentiment(symbol)
        
        # Machine Learning Prediction
        predictor = TradingPredictor()
        features = ['RSI', 'MACD', 'MACD_signal', 'Bollinger_high', 'Bollinger_low', 'SMA_20']
        ml_pred = predictor.predict(latest_data[features])[0]
        proba = predictor.predict_proba(latest_data[features])[0][ml_pred]
        ml_decision = "BUY" if ml_pred == 1 else "SELL"
        
        # Combine News and Social Sentiment for an overall sentiment decision
        sentiment_decision = decide_trade(news_sentiment, social_sentiment)
        
        # Reinforcement Learning Agent Decision
        rl_model, rl_env = load_rl_agent(df)
        obs = rl_env.reset()
        rl_action, _ = rl_model.predict(obs)
        rl_decision = ["HOLD", "BUY", "SELL"][rl_action]
        
        # Final combined decision: if all three agree, use it; otherwise, HOLD
        combined_decision = ml_decision if (ml_decision == sentiment_decision == rl_decision) else "HOLD"
        
        # Apply automated risk management (using an assumed entry price from 5 days ago)
        entry_price = df['Close'].iloc[-5]
        final_decision = risk_managed_decision(combined_decision, current_price, entry_price)
        
        # Display the live decision breakdown
        st.header("📈 Trading Decision")
        st.write(f"**Final Risk-Managed Decision:** {final_decision}")
        
        st.subheader("🔍 Decision Breakdown")
        st.write(f"- **ML Prediction:** {ml_decision} ({proba:.2%} confidence)")
        st.write(f"- **Sentiment Decision:** {sentiment_decision}")
        st.write(f"- **RL Agent Decision:** {rl_decision}")
        
        st.subheader("📝 Raw Sentiment Values")
        st.write(f"- **News Sentiment:** {news_sentiment:.4f}")
        st.write(f"- **Reddit Sentiment:** {social_sentiment:.4f}")
        
        st.subheader("📊 Latest Market Data")
        st.dataframe(df.tail())

# Backtesting Analysis Section
st.subheader("📊 Backtesting Analysis")
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        # Fetch historical data for 1 year for backtesting
        df_hist = fetch_stock_data(symbol, period='1y')
        # Run the backtest strategy (assumes 'SMA_20' is computed in df_hist)
        df_back, metrics = backtest_strategy(df_hist)
        
        st.write("### Performance Metrics")
        st.write(f"Cumulative Strategy Return: {metrics['cumulative_strategy_return']:.2%}")
        st.write(f"Cumulative Buy-and-Hold Return: {metrics['cumulative_buyhold_return']:.2%}")
        st.write(f"Sharpe Ratio (Strategy): {metrics['sharpe_ratio_strategy']:.2f}")
        st.write(f"Sharpe Ratio (Buy & Hold): {metrics['sharpe_ratio_buyhold']:.2f}")
        st.write(f"Maximum Drawdown (Strategy): {metrics['max_drawdown_strategy']:.2%}")
        
        # Plot cumulative returns for visual comparison
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_back.index, df_back['cumulative_strategy'], label='Strategy')
        ax.plot(df_back.index, df_back['cumulative_buyhold'], label='Buy & Hold', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        st.pyplot(fig)


st.subheader("Multi-Asset Backtesting")
tickers_input = st.text_input("Enter tickers (comma separated):", "AAPL,MSFT,GOOGL")
if st.button("Run Multi-Asset Backtest"):
    with st.spinner("Fetching multi-asset data and running simulation..."):
        tickers = [t.strip() for t in tickers_input.split(",")]
        data_dict = fetch_multi_asset_data(tickers, period='1y')
        portfolio = simulate_portfolio(data_dict, initial_capital=10000, window=20)
        
        st.subheader("Portfolio Value Over Time")
        st.line_chart(portfolio['portfolio_value'])
        
        # Optionally display the final portfolio value and return:
        final_value = portfolio['portfolio_value'].iloc[-1]
        pct_return = (final_value / 10000 - 1) * 100
        st.write(f"Final Portfolio Value: ${final_value:,.2f}")
        st.write(f"Overall Return: {pct_return:.2f}%")
