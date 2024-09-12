import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import spacy
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')

# Define relative paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

keywords_phrases = {
    "words": [
        "customer", "excellence", "obsession", "service", "easy",
        "simple", "performance", "decision making", "speed", "slow",
        "fast", "inclusion", "diversity", "collaboration", "speak",
        "belonging", "purpose", "behaviour",
    ],
    "phrases": [
        "customer excellence", "customer obsession", "decision making", "speak up", "customer service"
    ]
}

comments_data = pd.read_csv(os.path.join(data_dir, 'comments.csv'))
verbatim_data = pd.read_csv(os.path.join(data_dir, 'verbatim.csv'))

# Combine the datasets into a single DataFrame
combined_data = pd.concat([
    verbatim_data.rename(columns={'Response': 'Text'}),
    comments_data.rename(columns={'Comments': 'Text'})
], ignore_index=True)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming and lemmatization functions
def stem_and_lemmatize(text):
    words = text.split()
    stemmed_lemmatized_words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
    return ' '.join(stemmed_lemmatized_words)

# Apply stemming and lemmatization to keywords and phrases
keywords_phrases['stemmed_words'] = [stem_and_lemmatize(word) for word in keywords_phrases['words']]
keywords_phrases['stemmed_phrases'] = [stem_and_lemmatize(phrase) for phrase in keywords_phrases['phrases']]

# Ensure the 'Text' column is in the correct format
combined_data['OriginalText'] = combined_data['Text']
combined_data['Text'] = combined_data['Text'].astype(str).str.lower()

# Apply stemming and lemmatization to the combined data
combined_data['ProcessedText'] = combined_data['Text'].apply(stem_and_lemmatize)

# Function to check if any keyword or phrase is present in the text using fuzzy matching
def contains_keyword_or_phrase(text):
    for word in keywords_phrases['stemmed_words']:
        if fuzz.partial_ratio(word, text) > 80:
            return True
    for phrase in keywords_phrases['stemmed_phrases']:
        if fuzz.partial_ratio(phrase, text) > 80:
            return True
    return False

# Filter the combined data to keep only rows that contain any keyword or phrase
filtered_data = combined_data[combined_data['ProcessedText'].apply(contains_keyword_or_phrase)]

# Filter out rows with text below a certain length (e.g., 20 characters)
filtered_data = filtered_data[filtered_data['Text'].str.len() >= 20]

# Print the head of the filtered DataFrame to verify
print("Filtered DataFrame Head:")
print(filtered_data.head())

# Count the occurrences of each keyword and phrase in the filtered dataset
keyword_counts = Counter()
phrase_counts = Counter()

for text in filtered_data['Text']:
    words = text.split()
    for word in keywords_phrases['words']:
        if word in words:
            keyword_counts[word] += 1
    for phrase in keywords_phrases['phrases']:
        if phrase in text:
            phrase_counts[phrase] += 1

# Print the counts of keywords and phrases
print("\nKeyword Counts:")
print(keyword_counts)
print("\nPhrase Counts:")
print(phrase_counts)

# Visualization of keyword counts
plt.figure(figsize=(10, 6))
plt.bar(keyword_counts.keys(), keyword_counts.values())
plt.title('Keyword Counts')
plt.xticks(rotation=45)
plt.show()

# Visualization of phrase counts
plt.figure(figsize=(10, 6))
plt.bar(phrase_counts.keys(), phrase_counts.values())
plt.title('Phrase Counts')
plt.xticks(rotation=45)
plt.show()

# Generate a word cloud for the filtered text data
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_data['Text']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Filtered Feedback Text')
plt.show()

# Vectorize the text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['Text'])

# Apply K-means clustering
num_clusters = 5  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_data['Cluster'] = kmeans.fit_predict(tfidf_matrix)

# Print the number of items in each cluster
print(filtered_data['Cluster'].value_counts())

# PCA for dimensionality reduction (for visualization purposes)
pca = PCA(n_components=2, random_state=42)
reduced_tfidf_matrix = pca.fit_transform(tfidf_matrix.toarray())

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    points = reduced_tfidf_matrix[filtered_data['Cluster'] == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
plt.title('Scatter Plot of Clusters')
plt.legend()
plt.show()

# Save the filtered data with clusters to a new CSV file
filtered_data.to_csv(os.path.join(output_dir, 'filtered_combined_data.csv'), index=False)

# Function to get top terms in each cluster
def get_top_terms_per_cluster(tfidf_matrix, labels, terms, num_terms=10):
    cluster_terms = {}
    for cluster_num in np.unique(labels):
        cluster_indices = np.where(labels == cluster_num)
        cluster_matrix = tfidf_matrix[cluster_indices]
        mean_tfidf = np.mean(cluster_matrix, axis=0)
        sorted_indices = np.argsort(mean_tfidf).flatten()[::-1]
        top_terms = [terms[i] for i in sorted_indices[:num_terms]]
        cluster_terms[cluster_num] = top_terms
    return cluster_terms

# Get feature names (terms)
terms = tfidf_vectorizer.get_feature_names_out()

# Get top terms per cluster
top_terms_per_cluster = get_top_terms_per_cluster(tfidf_matrix, filtered_data['Cluster'], terms)

# Print top terms for each cluster
for cluster_num, top_terms in top_terms_per_cluster.items():
    print(f"Cluster {cluster_num}: {', '.join(map(str, top_terms))}")

# Function to print representative texts for each cluster
def print_representative_texts(filtered_data, num_texts=5):
    for cluster_num in filtered_data['Cluster'].unique():
        print(f"\nCluster {cluster_num} Representative Texts:")
        cluster_texts = filtered_data[filtered_data['Cluster'] == cluster_num]['Text'].head(num_texts)
        for i, text in enumerate(cluster_texts, 1):
            print(f"{i}. {text}")

# Print representative texts for each cluster
print_representative_texts(filtered_data)

# Function to get the related keywords or phrases in the text
def get_related_keywords_phrases(text):
    related = []
    for word in keywords_phrases['words']:
        if word in text:
            related.append(word)
    for phrase in keywords_phrases['phrases']:
        if phrase in text:
            related.append(phrase)
    return ', '.join(related)

# Add a new column with related keywords or phrases
filtered_data['RelatedKeywordsPhrases'] = filtered_data['Text'].apply(get_related_keywords_phrases)

# Save the final data with the specified columns to a new CSV file
final_output = filtered_data[['OriginalText', 'Cluster', 'RelatedKeywordsPhrases']]
final_output.to_csv(os.path.join(output_dir, 'final_output.csv'), index=False)