from textblob import TextBlob
from newsapi import NewsApiClient
import praw

# News API setup
newsapi = NewsApiClient(api_key='163296e1a4fa4de0b050244e181e9507')

# Reddit API setup (use your exact credentials here)
reddit = praw.Reddit(
    client_id="LXcKMrigukq_jaojP4ObxQ",
    client_secret="DHBaPjJHpBW2eqTd1sT5b8Cib5Xj0A",
    user_agent="financial_bot by u/Dull-Present-1246",
    username="Dull-Present-1246",
    password="YousefFadiw123"  # <-- put your Reddit password here
)

# News sentiment analysis
def analyze_news_sentiment(keyword, articles_count=50):
    articles = newsapi.get_everything(q=keyword, language='en', sort_by='relevancy', page_size=articles_count)
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    return sum(sentiments) / len(sentiments) if sentiments else 0

# Reddit social sentiment analysis
def analyze_social_sentiment(keyword, subreddit_names=['stocks', 'investing', 'cryptocurrency'], num_posts=50):
    sentiments = []
    try:
        for subreddit_name in subreddit_names:
            subreddit = reddit.subreddit(subreddit_name)
            for post in subreddit.search(keyword, limit=num_posts):
                sentiments.append(TextBlob(post.title).sentiment.polarity)
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Reddit API Error: {e}")
        return 0  # Return neutral sentiment if Reddit fails
