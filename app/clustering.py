import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import logging

class Helper:
    @staticmethod
    def print_scrape_status(count):
        logging.info(f'Scraped {count} articles so far...')

    @staticmethod
    def clean_dataframe(news_df):
        logging.info("Cleaning dataframe")
        news_df = news_df[news_df.title != '']
        news_df = news_df[news_df.body != '']
        news_df = news_df[news_df.image_url != '']
        news_df = news_df[news_df.title.str.count(r'\s+').ge(3)]
        news_df = news_df[news_df.body.str.count(r'\s+').ge(20)]
        return news_df

def compute_tfidf(news_df):
    logging.info("Computing TF-IDF values")
    tfidf_matrix = TfidfVectorizer().fit_transform(news_df['clean_body'])
    tfidf_array = np.asarray(tfidf_matrix.todense())
    return tfidf_array

def find_featured_clusters(clusters):
    logging.info("Finding clusters with articles from multiple sources")
    featured_clusters = {}
    for i in clusters.keys():
        if len(set([j["source"] for j in clusters[i]])) > 1:
            featured_clusters[i] = clusters[i]
    return featured_clusters
