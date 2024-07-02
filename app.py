import feedparser as fp
import dateutil.parser
from newspaper import Article, Config
import logging
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os
import random
import streamlit as st

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Helper:
    @staticmethod
    def print_scrape_status(count):
        logging.info(f'Scraped {count} articles so far...')

    @staticmethod
    def clean_dataframe(news_df):
        news_df = news_df[news_df.title != '']
        news_df = news_df[news_df.body != '']
        news_df = news_df[news_df.image_url != '']
        news_df = news_df[news_df.title.str.count('\s+').ge(3)]
        news_df = news_df[news_df.body.str.count('\s+').ge(20)]
        return news_df
    
    @staticmethod
    def clean_articles(news_df):
        news_df = (news_df.drop_duplicates(subset=["title", "source"])).sort_index()
        news_df = (news_df.drop_duplicates(subset=["body"])).sort_index()
        news_df = (news_df.drop_duplicates(subset=["url"])).sort_index()
        news_df = news_df.reset_index(drop=True)
        
        news_df['clean_body'] = news_df['body'].str.lower()
        
        stop_words = set(stopwords.words('english'))
        news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
        news_df['clean_body'] = news_df['clean_body'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        
        news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        
        logging.info("Contents of 'clean_body' after removing digits:")
        for i, body in enumerate(news_df['clean_body'].head(10)):
            logging.info(f'Article {i + 1}: {body}')
        
        sources_set = [x.lower() for x in set(news_df['source'])]
        sources_to_replace = dict.fromkeys(sources_set, "")
        news_df['clean_body'] = (news_df['clean_body'].replace(sources_to_replace, regex=True))
        
        news_df['clean_body'] = news_df['clean_body'].apply(unidecode)
        
        news_df['clean_body'] = news_df['clean_body'].apply(word_tokenize)
        
        stemmer = SnowballStemmer(language='english')
        news_df['clean_body'] = news_df['clean_body'].apply(lambda x: [stemmer.stem(y) for y in x])
        news_df['clean_body'] = news_df['clean_body'].apply(lambda x: ' '.join([word for word in x]))
        
        logging.info("Contents of 'clean_body' after cleaning:")
        for i, body in enumerate(news_df['clean_body'].head(10)):
            logging.info(f'Article {i + 1}: {body}')
        
        return news_df

    @staticmethod
    def shuffle_content(clusters_dict):
        for cluster in clusters_dict.values():
            random.shuffle(cluster)

    @staticmethod
    def prettify_similar(clusters_dict):
        similar_articles = []
        for cluster in clusters_dict.values():
            cluster_titles = [article['title'] for article in cluster]
            similar_articles.append(', '.join(cluster_titles))
        return similar_articles

def compute_tfidf(news_df):
    tfidf_matrix = TfidfVectorizer().fit_transform(news_df['clean_body'])
    tfidf_array = np.asarray(tfidf_matrix.todense())
    return tfidf_array

def find_featured_clusters(clusters):
    featured_clusters = {}
    for i in clusters.keys():
        if len(set([j["source"] for j in clusters[i]])) > 1:
            featured_clusters[i] = clusters[i]
    return featured_clusters

config = Config()
custom_tmp_dir = 'C:\\Users\\fredd\\custom_newspaper_tmp'

if not os.path.exists(custom_tmp_dir):
    os.makedirs(custom_tmp_dir)

config.fetch_images = False
config.memoize_articles = False
config.request_timeout = 10
config.directory = custom_tmp_dir

class CacheManager:
    def __init__(self, cache_file='article_cache.json'):
        self.cache_file = cache_file
        self.load_cache()
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    
    def get_article(self, url):
        return self.cache.get(url, None)
    
    def add_article(self, url, article_data):
        self.cache[url] = article_data
        self.save_cache()

class Scraper:
    def __init__(self, sources, days, cache_manager):
        self.sources = sources
        self.days = days
        self.cache_manager = cache_manager
    
    def scrape(self):
        try:
            articles_list = []
            now = datetime.now(timezone.utc)
            for source, content in self.sources.items():
                logging.info(f'Source: {source}')
                logging.info(f'Content: {content}')
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
                                Helper.print_scrape_status(len(articles_list))
                                continue
                            
                            try:
                                logging.info(f'Processing article: {entry.link}')
                                content = Article(entry.link, config=config)
                                content.download()
                                content.parse()
                                content.nlp()
                                try:
                                    article = {
                                        'source': source,
                                        'url': entry.link,
                                        'date': article_date.strftime('%Y-%m-%d'),
                                        'time': article_date.strftime('%H:%M:%S %Z'),
                                        'title': content.title,
                                        'body': content.text,
                                        'summary': content.summary,
                                        'keywords': content.keywords,
                                        'image_url': content.top_image
                                    }
                                    articles_list.append(article)
                                    Helper.print_scrape_status(len(articles_list))
                                    self.cache_manager.add_article(entry.link, article)
                                except Exception as e:
                                    logging.error(f'Error processing article: {e}')
                                    logging.info('Continuing...')
                            except Exception as e:
                                logging.error(f'Error downloading/parsing article: {e}')
                                logging.info('Continuing...')
            return articles_list
        except Exception as e:
            logging.error(f'Error in "Scraper.scrape()": {e}')
            raise Exception(f'Error in "Scraper.scrape()": {e}')

def main():
    st.title("News Scraper and Clustering App")
    
    sources_file = st.file_uploader("Upload your sources.json file", type="json")
    
    if sources_file is not None:
        sources = json.load(sources_file)
        
        days_to_scrape = st.slider("Days to scrape", 1, 30, 7)
        
        cache_manager = CacheManager()
        
        scraper = Scraper(sources, days_to_scrape, cache_manager)
        try:
            articles = scraper.scrape()
            
            if not articles:
                st.warning('No articles were scraped.')
            else:
                news_df = pd.DataFrame(articles)
                news_df = Helper.clean_dataframe(news_df)
                news_df = Helper.clean_articles(news_df)

                st.dataframe(news_df)

                tfidf_df = compute_tfidf(news_df)
                
                distance_threshold = st.slider("Distance threshold for clustering", 0.1, 2.0, 1.0)
                ac = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None).fit(tfidf_df)
                articles_labeled = ac.fit_predict(tfidf_df)
                
                clusters = {n: news_df.iloc[np.where(articles_labeled == n)].to_dict(orient='records') for n in np.unique(articles_labeled)}
                
                featured_clusters = find_featured_clusters(clusters)
                
                Helper.shuffle_content(featured_clusters)
                
                st.write("Featured Clusters:")
                for i, cluster in enumerate(featured_clusters.values()):
                    st.write(f"Cluster {i+1}:")
                    for article in cluster:
                        st.write(f"- {article['title']} ({article['source']})")
                
        except Exception as e:
            st.error(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
