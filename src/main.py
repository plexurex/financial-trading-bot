from utils.data_collection import fetch_stock_data
from utils.social_sentiment import analyze_news_sentiment, analyze_social_sentiment
from utils.trading_strategy import decide_trade

def main(symbol):
    data = fetch_stock_data(symbol)
    news_sentiment = analyze_news_sentiment(symbol)
    social_sentiment = analyze_social_sentiment(symbol)

    action = decide_trade(news_sentiment, social_sentiment)
    print(f"Decision for {symbol}: {action}")

if __name__ == "__main__":
    symbol = "AAPL"
    main(symbol)
