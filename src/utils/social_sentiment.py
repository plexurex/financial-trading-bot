
import feedparser
from textblob import TextBlob
import praw
from datetime import datetime, timedelta
import urllib.parse

# Profanity filter for Reddit titles
BANNED_WORDS = {"shit", "damn", "crap", "bollocks"}

# Reddit API setup
reddit = praw.Reddit(
    client_id="LXcKMrigukq_jaojP4ObxQ",
    client_secret="DHBaPjJHpBW2eqTd1sT5b8Cib5Xj0A",
    user_agent="financial_bot by u/Dull-Present-1246",
    username="Dull-Present-1246",
    password="YousefFadiw123"
)

def _clean_symbol(sym: str) -> str:
    """
    Convert "BTC-USD" → "BTC", etc., for subreddit searches.
    """
    return sym.split("-")[0]

#News Sentiment
def analyze_news_sentiment(keyword: str,
                           horizon_days: int = 1,
                           articles_count: int = 50) -> float:
    """
    Fetch up to `articles_count` headlines from
    Google News RSS search for `keyword` over the last `horizon_days`,
    then return average TextBlob polarity.
    """
    
    days = max(1, min(horizon_days, 365))
    #    # Google News RSS search
    q = urllib.parse.quote_plus(keyword)
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={q}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )
#    # Parse the RSS feed
    feed = feedparser.parse(rss_url)
    entries = feed.entries or []
    # take up to our limit
    titles = [e.title for e in entries[:articles_count] if hasattr(e, "title")]

    # sentiment
    polarities = [TextBlob(t).sentiment.polarity for t in titles]
    if not polarities:
        return 0.0
    return float(sum(polarities) / len(polarities))

# Social Sentiment
def analyze_social_sentiment(keyword: str,
                             horizon_days: int = 1,
                             subreddit_names: tuple[str, ...] = ('stocks', 'investing', 'cryptocurrency'),
                             num_posts: int = 50) -> float:
    """
    Pull up to `num_posts` titles from each subreddit, filtered by `horizon_days`:
      ≤1   → 'day'
      ≤30  → 'week'
      ≤60  → 'month'
      else → 'year'
    Filters out any containing profanity, then returns average polarity.
    """
    coin = _clean_symbol(keyword)

    if horizon_days <= 1:
        tf = 'day'
    elif horizon_days <= 30:
        tf = 'week'
    elif horizon_days <= 60:
        tf = 'month'
    else:
        tf = 'year'

    sentiments = []
    for name in subreddit_names:
        try:
            for post in reddit.subreddit(name).search(
                coin,
                limit=num_posts,
                time_filter=tf
            ):
                title = post.title.lower()
                if any(bad in title for bad in BANNED_WORDS):
                    continue
                sentiments.append(TextBlob(post.title).sentiment.polarity)
        except Exception:
            continue

    if not sentiments:
        return 0.0
    return float(sum(sentiments) / len(sentiments))
