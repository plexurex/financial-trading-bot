import streamlit as st
import pandas as pd
import numpy as np
import datetime
from textblob import TextBlob
from wordcloud import WordCloud
from pytrends.request import TrendReq
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px

from src.utils.data_collection        import fetch_stock_data
from src.models.multi_asset_trading  import fetch_multi_asset_data, simulate_portfolio
from src.utils.social_sentiment      import analyze_news_sentiment, analyze_social_sentiment, reddit
from src.utils.trading_strategy      import decide_trade, risk_managed_decision
from src.models.predictive_model     import MultiHorizonPredictor
from src.models.rl_agent            import load_rl_agent
from src.utils.backtesting           import backtest_strategy

st.set_page_config(
    page_title="QuantTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS styling (dark-mode cards) ---
st.markdown("""
<style>
  .main-title      {font-size:2.5rem!important; color:#1E88E5; text-align:center;}
  .section-header  {font-size:1.5rem; color:#DDD; border-bottom:2px solid #444; padding-bottom:4px;}
  .metric-card     {background-color:#222 !important; color:#EEE; border-radius:8px; padding:12px; margin:8px 0;}
  .metric-label    {color:#AAA;}
  .metric-value    {font-weight:bold;}
  .buy-card        {border-left:5px solid #36B37E !important;}
  .sell-card       {border-left:5px solid #FF5630 !important;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar navigation ---
st.sidebar.title("📊 QuantTrader Pro")
page = st.sidebar.radio("Navigate", ["Market Analysis","Backtesting","Portfolio Optimization"])
auto_refresh = st.sidebar.checkbox("Auto-refresh (1m)", value=True)
if auto_refresh:
    st_autorefresh(interval=60_000, key="refresh")
st.sidebar.markdown(f"**Last update:** {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

# --- Constants ---
crypto_list = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD",
               "SOL-USD","DOT-USD","MATIC-USD","AVAX-USD","LTC-USD"]
horizon_map  = {"Next Day":1,"1 Month":30,"2 Months":60,"1 Year":252}
hist_map     = {1:"90d",30:"365d",60:"730d",252:"1095d"}

def compute_efficient_frontier(mu: pd.Series,
                               cov: pd.DataFrame,
                               num_portfolios: int = 5000):
    """
    Approximate the efficient frontier by randomly sampling portfolios.
    Returns a DataFrame with columns:
      - 'vol'   : annualized portfolio volatility
      - 'ret'   : annualized portfolio return
      - 'weights': the weight vector (as a list) that achieved that point
    """
    n = len(mu)
    results = []
    for _ in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        port_ret = w.dot(mu)
        port_vol = np.sqrt(w @ cov.values @ w)
        results.append((port_vol, port_ret, w.copy()))
    ef = pd.DataFrame(results, columns=["vol", "ret", "weights"])
    ef = ef.sort_values("vol")
    frontier = []
    max_ret = -np.inf
    for vol, ret, w in ef.itertuples(index=False):
        if ret > max_ret:
            frontier.append((vol, ret, w))
            max_ret = ret
    return pd.DataFrame(frontier, columns=["vol", "ret", "weights"])

def solve_minvar_with_return(cov: pd.DataFrame,
                             mu: pd.Series,
                             target_ret: float):
    """
    Closed-form solution to:
      minimize w^T C w
      s.t. w^T 1 = 1, w^T mu = target_ret
    """
    invC = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    a = ones @ invC @ ones
    b = ones @ invC @ mu.values
    c = mu.values @ invC @ mu.values
    d = a*c - b*b
    lam =  (c - b*target_ret) / d
    gam = (a*target_ret - b) / d
    w = lam * (invC @ ones) + gam * (invC @ mu.values)
    return w

# --- MARKET ANALYSIS ---
if page == "Market Analysis":
    st.markdown("<h1 class='main-title'>Market Analysis & Trading Signals</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        symbol = st.selectbox("Select Crypto", crypto_list)
    with c2:
        horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()))
    with c3:
        st.write(""); st.write("")
        if st.button("Analyze & Trade", key="analyze"):
            st.session_state.run_analysis = True

    if st.session_state.get("run_analysis"):
        period = hist_map[horizon_map[horizon_label]]
        df     = fetch_stock_data(symbol, period=period, interval="1d")
        if df.empty:
            st.error("No data returned."); st.stop()

        latest = df.iloc[-1:]
        price  = float(latest["Close"].iloc[0])
        news_s = analyze_news_sentiment(symbol)
        soc_s  = analyze_social_sentiment(symbol)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                 line=dict(color="#1E88E5", width=2)))
        if "SMA_20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA20",
                                     line=dict(color="#FF9800", dash="dash")))
        if "SMA_50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA50",
                                     line=dict(color="#4CAF50", dash="dot")))

        predictor = MultiHorizonPredictor(symbol)
        hdays      = horizon_map[horizon_label]
        feats      = ["RSI","MACD","MACD_signal","Bollinger_high","Bollinger_low","SMA_20"]
        pred_class, proba = predictor.predict(latest[feats], hdays)
        hret    = df["Close"].pct_change(periods=hdays).dropna()
        med_ret = hret.median() if not hret.empty else df["Close"].pct_change().median()
        shift   = np.clip((proba - 0.5)*2*med_ret, -abs(med_ret)*2, abs(med_ret)*2)
        fut_pr  = price*(1+shift)
        last_dt = df.index[-1]
        fut_dt  = last_dt + pd.Timedelta(days=hdays)
        fig.add_trace(go.Scatter(x=[last_dt,fut_dt],
                                 y=[price, fut_pr],
                                 mode="lines+markers",
                                 name="ML Forecast",
                                 line=dict(color="magenta", dash="dot")))

        fig.update_layout(
            title=f"{symbol} Price & {horizon_label} Forecast",
            template="plotly_dark",
            height=450,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        rl_model, rl_env = load_rl_agent(df)
        obs             = rl_env.reset()
        rl_act, _       = rl_model.predict(obs)
        action = (int(rl_act.item())
                  if hasattr(rl_act, "item")
                  else int(np.atleast_1d(rl_act)[0]))
        rl_dec   = ["HOLD","BUY","SELL"][action]
        sent_dec = decide_trade(news_s, soc_s)
        final    = "HOLD"
        if pred_class == 1 and rl_dec == "BUY" and sent_dec == "BUY":
            final = "BUY"
        if pred_class == 0 and rl_dec == "SELL" and sent_dec == "SELL":
            final = "SELL"
        final = risk_managed_decision(final, price,
                                      df["Close"].iloc[-5] if len(df)>=5 else price)

        color = "#36B37E" if final=="BUY" else "#FF5630" if final=="SELL" else "#6554C0"
        st.markdown(f"""
        <div class="metric-card {'buy-card' if final=='BUY' else 'sell-card' if final=='SELL' else ''}">
          <p class="metric-label">Recommended Action</p>
          <p class="metric-value" style="color:{color}; font-size:1.8rem;">{final}</p>
          <p class="metric-label">Current Price</p>
          <p class="metric-value">${price:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Decision Factors"):
            st.write(f"- ML ({horizon_label}): {'UP' if pred_class==1 else 'DOWN'} ({proba:.1%})")
            st.write(f"- RL Agent: {rl_dec}")
            st.write(f"- Sentiment: {sent_dec}")
            st.write(f"- News Sentiment: {news_s:.4f}")
            st.write(f"- Social Sentiment: {soc_s:.4f}")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=(news_s+soc_s)/2,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Overall Sentiment"},
            gauge={
                'axis':{'range':[-1,1]},
                'steps':[
                  {'range':[-1,-0.3],'color':"#FF5252"},
                  {'range':[-0.3,0.3],'color':"#FFD740"},
                  {'range':[0.3,1],'color':"#4CAF50"}],
                'bar':{'color':"#1E88E5"}
            }
        ))
        gauge.update_layout(template="plotly_dark", height=300)

        g1, g2 = st.columns([2,1])
        with g1:
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown("""
            **Sentiment Score Guide**  
            - **[-1.0 … –0.3]** Negative  
            - **(-0.3 … 0.3)** Neutral  
            - **[0.3 … +1.0]** Positive
            """)
        with g2:
            posts = [p.title for p in reddit.subreddit("wallstreetbets").search(symbol, limit=30)]
            if posts:
                wc = WordCloud(background_color="white").generate(" ".join(posts))
                st.image(wc.to_array(), use_container_width=True)
            try:
                py = TrendReq(); py.build_payload([symbol], timeframe="now 12-H")
                tr = py.interest_over_time()[symbol]
                fig2 = px.line(tr, labels={"value":"Trend","date":"Time"})
                fig2.update_layout(template="plotly_dark", height=200)
                st.plotly_chart(fig2, use_container_width=True)
            except:
                pass

# --- BACKTESTING ---
elif page == "Backtesting":
    st.markdown("<h1 class='main-title'>Strategy Backtesting</h1>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Single-Asset","Multi-Asset"])

    with t1:
        c1,c2,c3 = st.columns([2,2,1])
        with c1:
            sym = st.selectbox("Crypto", crypto_list, key="bt_sym")
        with c2:
            per = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=3)
        with c3:
            st.write("")
            st.write("")
            run_bt = st.button("Run Backtest")
        if run_bt:
            dfh, met = backtest_strategy(fetch_stock_data(sym, period=per, interval="1d"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dfh.index, y=dfh["cumulative_strategy"], name="Strategy"))
            fig.add_trace(go.Scatter(x=dfh.index, y=dfh["cumulative_buyhold"],
                                     name="Buy&Hold", line=dict(dash="dash")))
            fig.update_layout(
                title=f"{sym} Cumulative Returns",
                template="plotly_dark",
                height=600,
                margin=dict(l=20,r=20,t=40,b=20)
            )
            ch, metcol = st.columns([3,1])
            with ch:   st.plotly_chart(fig, use_container_width=True)
            with metcol:
                st.markdown("### Metrics")
                st.markdown(f"- Strat Return: {met['cumulative_strategy_return']:.2%}")
                st.markdown(f"- B&H Return: {met['cumulative_buyhold_return']:.2%}")
                st.markdown(f"- Sharpe: {met['sharpe_ratio_strategy']:.2f}")
                st.markdown(f"- Max Drawdown: {met['max_drawdown_strategy']:.2%}")

    with t2:
        tickers = st.multiselect("Select up to 3", crypto_list,
                                 default=crypto_list[:3], max_selections=3)
        if st.button("Run Portfolio Backtest"):
            data_dict = {t: fetch_stock_data(t, period="1y", interval="1d")
                         for t in tickers}
            port = simulate_portfolio(data_dict, initial_capital=10000, window=20)
            st.line_chart(port["portfolio_value"], use_container_width=True)
            final = port["portfolio_value"].iloc[-1]
            st.markdown(f"**Final Value:** ${final:,.2f}  **Return:** {(final/10000-1)*100:.2f}%")

# --- PORTFOLIO OPTIMIZATION ---
elif page == "Portfolio Optimization":
    st.markdown("<h1 class='main-title'>Portfolio Optimization</h1>", unsafe_allow_html=True)

    assets = st.multiselect("Select Cryptos for Optimization",
                             crypto_list, default=crypto_list[:5])
    mode = st.radio("Optimization Mode", ["Minimize Variance", "Target Return"])
    if mode == "Target Return":
        target_ret = st.slider("Target Annual Return (%)",
                                0.0, 200.0, 50.0)/100.0
    else:
        risk_aversion = st.slider("Risk Aversion (λ)",
                                  0.0, 10.0, 1.0)

    if st.button("Run Optimization"):
        price_df = pd.concat([
            fetch_stock_data(sym, period="1y", interval="1d")["Close"].rename(sym)
            for sym in assets
        ], axis=1).dropna()
        rets = price_df.pct_change().dropna()
        mu   = rets.mean() * 252
        cov  = rets.cov() * 252

        if mode == "Minimize Variance":
            invC = np.linalg.inv(cov.values)
            w = invC.dot(np.ones(len(mu)))
            w /= w.sum()
        else:
            w = solve_minvar_with_return(cov, mu, target_ret)

        weights = pd.Series(w, index=mu.index)
        st.table(weights.to_frame("Weight").style.format("{:.1%}"))

        ef = compute_efficient_frontier(mu, cov)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ef["vol"], y=ef["ret"],
                                 mode="lines", name="Efficient Frontier"))
        fig.add_trace(go.Scatter(x=[np.sqrt(w.T @ cov.values @ w)],
                                 y=[w.dot(mu)],
                                 mode="markers", name="Your Portfolio",
                                 marker=dict(size=12, color="#36B37E")))
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Annual Volatility",
            yaxis_title="Annual Return",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        wealth = (rets.dot(w) + 1).cumprod()
        st.line_chart(wealth, use_container_width=True)
