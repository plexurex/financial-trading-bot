def decide_trade(news_sentiment, social_sentiment):
    combined_score = (news_sentiment + social_sentiment) / 2
    if combined_score > 0.2:
        return "BUY"
    elif combined_score < -0.2:
        return "SELL"
    else:
        return "HOLD"
