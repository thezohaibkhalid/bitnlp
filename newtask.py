import nltk
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Sample sentences
sentences = [
    "Lahore is a prominent city in Punjab, Pakistan, celebrated for its rich cultural heritage and textile manufacturing.",
    "The Institute of Lahore is located in the city center and offers a wide array of educational programs, including technology and healthcare."
]

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Step 1: Tokenization
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

# Step 2: Stop Word Removal
stop_words = set(stopwords.words('english'))
filtered_sentences = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences]

# Step 3: POS Tagging
pos_tagged_sentences = [pos_tag(sentence) for sentence in filtered_sentences]

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed_sentences = [[stemmer.stem(word) for word, tag in sentence] for sentence in pos_tagged_sentences]

# Step 5: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_sentences = [[lemmatizer.lemmatize(word) for word, tag in sentence] for sentence in pos_tagged_sentences]

# Step 6: Chunking
chunked_sentences = [ne_chunk(sentence) for sentence in pos_tagged_sentences]

# Step 7: Parse Tree Generation
def draw_parse_tree(tree):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.show()

# Create and draw parse trees
for sentence in pos_tagged_sentences:
    tree = ne_chunk(sentence)
    draw_parse_tree(tree)

# Step 8: Bag of Words
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform([' '.join(sentence) for sentence in filtered_sentences]).toarray()
print("Bag of Words:")
print(X_bow)

# Step 9: TF-IDF Calculation
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform([' '.join(sentence) for sentence in filtered_sentences]).toarray()
print("TF-IDF:")
print(X_tfidf)
