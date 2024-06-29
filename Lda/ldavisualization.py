import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import preprocess_string
import pyLDAvis
import pyLDAvis.gensim_models

# Load the CSV file
file_path = 'path_to_your/cleaned_news_articles.csv'
data = pd.read_csv(file_path)

# Custom preprocessing function
def preprocess(text):
    text = preprocess_string(text.lower())
    text = [word for word in text if word not in ENGLISH_STOP_WORDS and word.isalpha()]
    return text

# Apply preprocessing
texts = data['clean_body'].dropna().tolist()
processed_texts = [preprocess(text) for text in texts]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Fit the LDA model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42, passes=10)

# Prepare the visualization
lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)

# Save the visualization to an HTML file
pyLDAvis.save_html(lda_display, 'lda_visualization.html')

# Display the visualization
pyLDAvis.show(lda_display)
