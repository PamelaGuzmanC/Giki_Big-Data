import feedparser as fp
import dateutil.parser
from newspaper import Article, Config
import logging
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from textblob import TextBlob
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from nltk.tokenize import word_tokenize
from unidecode import unidecode

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    def __init__(self, cache_file='article_cache.json'):
        self.cache_file = cache_file
        self.load_cache()
    
    def load_cache(self):
        logging.info("Loading cache")
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
    
    def save_cache(self):
        logging.info("Saving cache")
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    
    def get_article(self, url):
        return self.cache.get(url, None)
    
    def add_article(self, url, article_data):
        logging.info(f'Adding article to cache: {url}')
        self.cache[url] = article_data
        self.save_cache()

class Scraper:
    def __init__(self, sources, days, cache_manager):
        self.sources = sources
        self.days = days
        self.cache_manager = cache_manager
    
    def scrape(self):
        articles_list = []
        new_articles_count = 0
        now = datetime.now(timezone.utc)
        for source, content in self.sources.items():
            logging.info(f'Source: {source}')
            for url in content['rss']:
                logging.info(f'Processing RSS feed: {url}')
                try:
                    d = fp.parse(url)
                except Exception as e:
                    logging.error(f'Error parsing RSS feed {url}: {e}')
                    continue
                
                for entry in d.entries:
                    if not hasattr(entry, 'published'):
                        logging.warning(f'Entry missing "published" attribute: {entry}')
                        continue
                    
                    try:
                        article_date = dateutil.parser.parse(getattr(entry, 'published'))
                        article_date = article_date.astimezone(timezone.utc)
                        logging.info(f'Found article with date: {article_date}')
                    except Exception as e:
                        logging.error(f'Error parsing article date: {e}')
                        continue
                    
                    if now - article_date <= timedelta(days=self.days):
                        cached_article = self.cache_manager.get_article(entry.link)
                        if cached_article:
                            logging.info(f'Using cached article: {entry.link}')
                            articles_list.append(cached_article)
                            continue
                        
                        try:
                            logging.info(f'Processing article: {entry.link}')
                            content = Article(entry.link, config=config)
                            content.download()
                            content.parse()
                            content.nlp()
                            try:
                                sentiment = TextBlob(content.text).sentiment.polarity
                                sentiment_category = 'positive' if sentiment > 0 else 'neutral' if sentiment == 0 else 'bad'
                                
                                article = {
                                    'source': source,
                                    'url': entry.link,
                                    'date': article_date.strftime('%Y-%m-%d'),
                                    'time': article_date.strftime('%H:%M:%S %Z'),
                                    'title': content.title,
                                    'body': content.text,
                                    'summary': content.summary,
                                    'keywords': content.keywords,
                                    'image_url': content.top_image,
                                    'sentiment': sentiment,
                                    'sentiment_category': sentiment_category
                                }
                                
                                articles_list.append(article)
                                self.cache_manager.add_article(entry.link, article)
                                new_articles_count += 1
                            except Exception as e:
                                logging.error(f'Error processing article: {e}')
                                logging.info('Continuing...')
                        except Exception as e:
                            logging.error(f'Error downloading/parsing article: {e}')
                            logging.info('Continuing...')
        logging.info(f'Total new articles scraped: {new_articles_count}')
        return articles_list

def clean_articles(news_df):
    news_df['clean_body'] = news_df['body'].str.lower()
    stop_words = set(stopwords.words('english'))
    news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    news_df['clean_body'] = news_df['clean_body'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    news_df['clean_body'] = news_df['clean_body'].apply(unidecode)
    news_df['clean_body'] = news_df['clean_body'].apply(word_tokenize)
    stemmer = SnowballStemmer(language='english')
    news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))

    return news_df

def sentiment_analysis(articles):
    logging.info("Performing sentiment analysis")
    
    articles_df = pd.DataFrame(articles)
    articles_df['sentiment'] = articles_df['body'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    def classify_sentiment(polarity):
        if polarity > 0:
            return 'positive'
        elif polarity == 0:
            return 'neutral'
        else:
            return 'bad'
    
    articles_df['sentiment_category'] = articles_df['sentiment'].apply(classify_sentiment)
    
    return articles_df[['url', 'sentiment', 'sentiment_category']]

# Custom configuration for the newspaper library
config = Config()
config.fetch_images = False
config.memoize_articles = False
config.request_timeout = 10

if __name__ == '__main__':
    logging.info("Starting main script")
    with open('app/sources.json', 'r') as file:
        sources = json.load(file)
    
    days_to_scrape = 7
    
    cache_manager = CacheManager()
    
    scraper = Scraper(sources, days_to_scrape, cache_manager)
    try:
        articles = scraper.scrape()
        
        if not articles:
            logging.warning('No articles were scraped.')
        else:
            logging.info(f'{len(articles)} articles scraped.')
            news_df = pd.DataFrame(articles)
            news_df = clean_articles(news_df)
            
            sentiment_df = sentiment_analysis(articles)
            
            news_df.drop(columns=['sentiment', 'sentiment_category'], inplace=True, errors='ignore')

            news_df = pd.merge(news_df, sentiment_df[['url', 'sentiment', 'sentiment_category']], on='url')
            
            # Save cleaned and analyzed articles to cache
            for article in news_df.to_dict(orient='records'):
                cache_manager.add_article(article['url'], article)
            
    except Exception as e:
        logging.error(f'An error occurred: {e}')
