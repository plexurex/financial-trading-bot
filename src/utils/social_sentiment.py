import tweepy
from textblob import TextBlob
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='163296e1a4fa4de0b050244e181e9507')

# Tweepy setup with your API credentials
auth = tweepy.OAuthHandler('1o9gRUlX2t5pM0GMUnlSulQe0', 'Ns9bLzOWDUxURb87E6hgiJRK3V7F6y5IGFFfDKWiMamNkPT8ib')
auth.set_access_token('746454958933942272-5f8F90clSsNhzpKxq2Hoe2n5SHkKx3A', '4Vm9VeMbdp11xTu9ClNyJV7mnvbYgAiuf7rtyZn2bwxl0')
api = tweepy.API(auth)

def analyze_news_sentiment(keyword, articles_count=50):
    articles = newsapi.get_everything(q=keyword, language='en', sort_by='relevancy', page_size=articles_count)
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    return sum(sentiments) / len(sentiments) if sentiments else 0

def analyze_social_sentiment(keyword, num_tweets=50):
    sentiments = []
    try:
        tweets = api.search_tweets(q=keyword, count=num_tweets, lang='en', tweet_mode='extended')
        for tweet in tweets:
            content = tweet.full_text
            sentiments.append(TextBlob(content).sentiment.polarity)
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Twitter API Error: {e}")
        return 0  # Default neutral sentiment if errors occur
