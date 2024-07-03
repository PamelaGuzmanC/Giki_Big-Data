import streamlit as st
import pandas as pd
import os
from clustering import compute_tfidf
from sklearn.cluster import AgglomerativeClustering
import json
import numpy as np

st.set_page_config(layout="wide")

ARTICLES_CACHE_FILE = 'article_cache.json'

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_articles_from_cache(cache_file):
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                articles = json.load(f)
            return pd.DataFrame(articles.values())
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return pd.DataFrame()

def filter_articles_by_keywords(articles, keywords):
    # Ensure keywords is a list
    if not isinstance(keywords, list):
        keywords = [keywords] if keywords else []

    filtered_articles = []
    for article in articles:
        body = article.get('body', '')
        if not isinstance(body, str):
            body = str(body)
        if any(keyword.lower() in body.lower() for keyword in keywords if keyword):
            filtered_articles.append(article)
    
    return filtered_articles

def cluster_articles(articles_df, keyword):
    # Ensure 'body' is a string and fill NaN values with empty strings
    articles_df['body'] = articles_df['body'].apply(lambda x: str(x) if not isinstance(x, str) else x).fillna('')

    if keyword:
        articles_df = articles_df[articles_df['body'].str.contains(keyword, case=False, na=False)]

    if articles_df.empty:
        return pd.DataFrame(), []

    # Ensure no NaN values in the DataFrame for clustering
    articles_df.fillna('', inplace=True)

    tfidf_df = compute_tfidf(articles_df)
    distance_threshold = 1.5
    ac = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
    articles_labeled = ac.fit_predict(tfidf_df)

    articles_df['cluster_id'] = articles_labeled
    clusters = {str(n): articles_df.iloc[np.where(articles_labeled == n)].to_dict(orient='records') for n in np.unique(articles_labeled)}

    return articles_df, clusters

def display_articles(articles_df, clusters, clusters_per_row=3):
    if articles_df.empty:
        st.write("No articles found with the given keyword.")
    else:
        # Group articles by clusters
        grouped = articles_df.groupby('cluster_id')
        
        cluster_ids = sorted(grouped.groups.keys())

        for i in range(0, len(cluster_ids), clusters_per_row):
            cluster_subset = cluster_ids[i:i+clusters_per_row]
            cols = st.columns(len(cluster_subset))


            for col, cluster_id in zip(cols, cluster_subset):
                with col:
                    st.markdown(f"## Cluster {cluster_id}")
                    group = grouped.get_group(cluster_id)
                    for idx, article in group.iterrows():
                        if article.get('image_url'):
                            st.image(article['image_url'], use_column_width=True)  # Make the image width dynamic

                        st.markdown(f"### [{article.get('title')}]({article.get('url')})")
                        st.subheader(f"Source: {article.get('source')}")
                        st.write(f"Published on: {article.get('date')} at {article.get('time')}")
                        st.write(article.get('summary'))
                        st.write(f"Keywords: {', '.join(article.get('keywords', []))}")
                        st.write(f"Sentiment: {article.get('sentiment_category')}")
                        st.write(f"Contains Keyword: {article.get('contains_keyword')}")
                        st.write(f"Cluster ID: {article.get('cluster_id')}")
                        st.write("---")

if __name__ == '__main__':
    st.title("News Articles")

    keyword = st.text_input("Search articles by keyword")
    articles_df = load_articles_from_cache(ARTICLES_CACHE_FILE)
    
    if not articles_df.empty:
        print("Loaded articles from cache successfully.")
    
    # Filter articles by keyword
    filtered_articles = filter_articles_by_keywords(articles_df.to_dict(orient='records'), [keyword])
    
    # Convert filtered articles back to DataFrame for clustering
    filtered_articles_df = pd.DataFrame(filtered_articles)
    
    filtered_articles_df, clusters = cluster_articles(filtered_articles_df, keyword)
    
    display_articles(filtered_articles_df, clusters)




