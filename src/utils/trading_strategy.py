def decide_trade(news_sentiment, social_sentiment):
    combined_score = (news_sentiment + social_sentiment) / 2
    if combined_score > 0.2:
        return "BUY"
    elif combined_score < -0.2:
        return "SELL"
    else:
        return "HOLD"

def decide_trade(news_sentiment, social_sentiment):
    combined_score = (news_sentiment + social_sentiment) / 2
    if combined_score > 0.2:
        return "BUY"
    elif combined_score < -0.2:
        return "SELL"
    else:
        return "HOLD"

def risk_managed_decision(predicted_decision, current_price, entry_price, stop_loss_pct=0.05, take_profit_pct=0.1):
    if predicted_decision == "BUY":
        if current_price <= entry_price * (1 - stop_loss_pct):
            return "SELL"
        elif current_price >= entry_price * (1 + take_profit_pct):
            return "SELL"
        else:
            return "BUY"
    elif predicted_decision == "SELL":
        return "SELL"
    else:
        return "HOLD"
