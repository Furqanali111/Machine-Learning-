import pandas as pd

# Read CSV file
df = pd.read_csv("D:/Work/Fiver/Ldgs/07/ldgs333-attachments/dataset_9.csv")
# Filter the dataset for the "action" or "comedy" genre
filtered_df = df[df['genre'].isin(['Action', 'Comedy'])]

# Group the dataset by genre and aggregate the reviews
grouped_df = filtered_df.groupby('genre')['review'].apply(list).reset_index()

# Access the reviews for action genre (if it exists)
if 'Action' in grouped_df['genre'].values:
    action_reviews = grouped_df[grouped_df['genre'] == 'Action']['review'].iloc[0]
else:
    action_reviews = []

# Access the reviews for comedy genre (if it exists)
if 'Comedy' in grouped_df['genre'].values:
    comedy_reviews = grouped_df[grouped_df['genre'] == 'Comedy']['review'].iloc[0]
else:
    comedy_reviews = []

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

import re


def preprocess_dataset(dataset):
    # Convert to lowercase
    dataset = dataset.lower()

    # Remove special characters and digits
    dataset = re.sub(r'[^a-zA-Z\s]', '', dataset)

    # Tokenize the dataset
    tokens = nltk.word_tokenize(dataset)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    preprocessed_dataset = ' '.join(tokens)

    return preprocessed_dataset

action_processed=[preprocess_dataset(review) for review in action_reviews]

from sklearn.feature_extraction.text import CountVectorizer


def create_document_term_matrix(dataset):
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Create the document-term matrix
    dtm = vectorizer.fit_transform(dataset)

    # Get the feature names (terms)
    feature_names = vectorizer.get_feature_names_out()

    return dtm, feature_names

# Create the document-term matrix
action_vectors, Feature_names = create_document_term_matrix(action_processed)

from sklearn.decomposition import LatentDirichletAllocation


def apply_lda(document_term_matrix, num_topics):
    # Initialize the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    # Fit the LDA model to the document-term matrix
    lda.fit(document_term_matrix)

    # Return the trained LDA model
    return lda

action_LDA=apply_lda(action_vectors,5)

import numpy as np


def extract_topics(lda_model, feature_names, num_words):
    # Get the topic-word distributions from the LDA model
    topic_word_distributions = lda_model.components_

    # Get the document-topic distributions from the LDA model
    document_topic_distributions = lda_model.transform(action_vectors)

    # Aggregate the topic prevalence across the corpus
    topic_prevalence = np.sum(document_topic_distributions, axis=0)

    # Sort the topics based on prevalence
    sorted_topics = np.argsort(topic_prevalence)[::-1]

    # Iterate over each topic
    for rank, topic_idx in enumerate(sorted_topics):
        # Get the top N words for the topic
        topic_words = topic_word_distributions[topic_idx]
        top_words_idx = topic_words.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[idx] for idx in top_words_idx]

        # Print the topic rank, prevalence, and top words
        print(f"Rank {rank + 1} | Topic Prevalence: {topic_prevalence[topic_idx]}")
        print(f"Top Words: {', '.join(top_words)}")
        print()
    return topic_prevalence

topic_prevalence=extract_topics(action_LDA,Feature_names,5)

print(topic_prevalence)

import numpy as np
import pyLDAvis.sklearn


import numpy as np
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer


def create_topic_map(topic_prevalence):
    # Create a dummy input dataset
    dataset = ['dummy']

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit the vectorizer on the dummy dataset to obtain the vocabulary
    vectorizer.fit_transform(dataset)

    # Create an empty document-term matrix with the number of topics as the number of rows
    document_term_matrix = np.zeros((len(topic_prevalence), len(vectorizer.get_feature_names_out())))

    # Convert topic prevalence to a numpy array
    topic_prevalence = np.array(topic_prevalence)

    # Reshape the topic prevalence array to match the document-term matrix shape
    topic_prevalence = topic_prevalence.reshape(-1, 1)

    # Create the topic map using pyLDAvis
    topic_map = pyLDAvis.sklearn.prepare(topic_prevalence, document_term_matrix, vectorizer, mds='mmds')

    # Return the topic map
    return topic_map

topic_prevalence = [12.35335932, 373.51908353, 591.16894142, 188.9458315, 103.01278423]
topic_map = create_topic_map(topic_prevalence)
pyLDAvis.display(topic_map)

