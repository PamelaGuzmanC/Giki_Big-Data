import streamlit as st
import pandas as pd
import os
import json

# Define CSS to ensure columns have the same height
css = """
<style>
    .stColumn {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .stColumn > div {
        height: 100%;
    }
</style>
"""

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_articles(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            articles = json.load(f)
        return pd.DataFrame(articles.values())
    else:
        return pd.DataFrame()

def display_articles(articles_df, keyword=None):
    if keyword:
        articles_df = articles_df[articles_df['body'].str.contains(keyword, case=False, na=False)]

    if articles_df.empty:
        st.write("No articles found with the given keyword.")
    else:
        cols = st.columns(2)  # Create two columns for layout
        for idx, article in articles_df.iterrows():
            col = cols[idx % 2]  # Alternate articles between columns
            with col:
                st.markdown(css, unsafe_allow_html=True)
                if article['image_url']:
                    st.image(article['image_url'], width=600)  # Set image width to medium size
                
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.subheader(f"Source: {article['source']}")
                st.write(f"Published on: {article['date']} at {article['time']}")
                st.write(article['summary'])
                st.write(f"Keywords: {', '.join(article['keywords'])}")
                st.write(f"Sentiment: {article['sentiment_category']}")
                st.write(f"Contains Keyword: {article['contains_keyword']}")
                st.write("---")

if __name__ == '__main__':
    st.title("News Articles")

    keyword = st.text_input("Search articles by keyword")
    articles_df = load_articles('article_cache.json')
    display_articles(articles_df, keyword)
