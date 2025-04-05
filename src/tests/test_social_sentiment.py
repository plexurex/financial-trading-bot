from src.utils.social_sentiment import analyze_social_sentiment

def test_reddit_sentiment():
    symbol = "BTC"
    sentiment = analyze_social_sentiment(symbol)
    print(f"Reddit sentiment for '{symbol}': {sentiment}")

if __name__ == "__main__":
    test_reddit_sentiment()
