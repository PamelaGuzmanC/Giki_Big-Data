import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Load the cached articles
with open('article_cache.json', 'r') as file:
    articles = json.load(file)

# Convert articles to a DataFrame
df = pd.DataFrame(articles.values())

st.set_page_config(layout="wide")

# Display the title and date
st.title("Daily News")
st.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

# Check if the DataFrame has a 'cluster_id' column
if 'cluster_id' not in df.columns:
    st.error("The 'cluster_id' column is missing from the dataset.")
else:
    # Group articles by cluster and display them
    for cluster_id, cluster_articles in df.groupby('cluster_id'):
        st.header(f"Cluster {cluster_id}")
        for _, article in cluster_articles.iterrows():
            st.subheader(article['title'])
            st.write(f"Source: {article['source']}")
            st.write(f"Summary: {article['summary']}")
            st.write(f"Sentiment: {article['sentiment_category_y']}")
            st.write(f"Date: {article['date']} {article['time']}")
            st.write(f"Keywords: {', '.join(article['keywords']) if isinstance(article['keywords'], list) else article['keywords']}")
            if article['image_url']:
                st.image(article['image_url'], width=400)
            st.markdown(f"[Read more]({article['url']})")

# Run this app by using the command: streamlit run app.py
