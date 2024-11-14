import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Input text examples
input_text_tokenization = "This is a sample text for tokenization."
input_text_segmentation = "This is the first sentence. Here is another one!"
input_text_lowercasing = "Sample Text for Lowercasing."
input_text_stopwords = "This is a sample sentence for stop word removal."
input_text_punctuation = "Text with 123 and some punctuation marks!."
input_text_stemming = "Text with various forms of words for stemming."
input_text_lemmatization = "This text is for lemmatization testing."
input_text_pos = "This is a sentence for POS tagging."
input_text_bow = "This is a sample sentence for bag of words calculation."
input_text_tfidf = "This is a sample text for TF-IDF calculation."
documents_for_embedding = ["This is the first document.", "And this is the second document."]

# Program 1: Tokenization
print("Tokenization:")
doc = nlp(input_text_tokenization)
tokens = [token.text for token in doc]
print("\n".join(tokens))
print("\n" + "-"*40 + "\n")

# Program 2: Segmentation (Sentence Splitting)
print("Segmentation:")
doc = nlp(input_text_segmentation)
sentences = [sent.text for sent in doc.sents]
print("\n".join(sentences))
print("\n" + "-"*40 + "\n")

# Program 3: Lowercasing
print("Lowercasing:")
doc = nlp(input_text_lowercasing)
lowercased_tokens = [token.text.lower() for token in doc]
print("\n".join(lowercased_tokens)) # Output: "this is a sample text for tokenization."
print("\n" + "-"*40 + "\n")

# Program 4: Stop Word Removal
print("Tokens after Stop Word Removal:")
doc = nlp(input_text_stopwords)
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print("\n".join(filtered_tokens))  
print("\n" + "-"*40 + "\n")

# Program 5: Removing Digits and Punctuation
print("Tokens after Removing Punctuation/Digits:")
doc = nlp(input_text_punctuation)
cleaned_tokens = [token.text for token in doc if token.is_alpha]
print("\n".join(cleaned_tokens))
print("\n" + "-"*40 + "\n")

# Program 6: Stemming (Using Lemmatization)
print("Stemming:")
doc = nlp(input_text_stemming)
stemmed_tokens = [token.lemma_ for token in doc if token.is_alpha]
print("\n".join(stemmed_tokens))
print("\n" + "-"*40 + "\n")

# Program 7: Lemmatization
print("Lemmatization:")
doc = nlp(input_text_lemmatization)
lemmatized_tokens = [token.lemma_ for token in doc]
print("\n".join(lemmatized_tokens))
print("\n" + "-"*40 + "\n")

# Program 8: POS Tagging
print("POS Tagging:")
doc = nlp(input_text_pos)
pos_tags = [f"{token.text} ({token.pos_})" for token in doc]
print("\n".join(pos_tags))
print("\n" + "-"*40 + "\n")

# Program 9: Bag of Words
print("Bag of Words:")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([input_text_bow])
words = vectorizer.get_feature_names_out()
vector = X.toarray().flatten()
for word, count in zip(words, vector):
    print(f"{word}: {count}")
print("\nBag of Words Vector Representation:", vector)
print("\n" + "-"*40 + "\n")

# Program 10: Word Embedding
print("Word Embedding:")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents_for_embedding)
feature_names = vectorizer.get_feature_names_out()
word = 'document'
if word in feature_names:
    word_index = np.where(feature_names == word)[0][0]
    word_vector = X[:, word_index].toarray().flatten()
    print(f"Word Embedding for '{word}': {word_vector}")
else:
    print(f"'{word}' not in vocabulary.")
print("\n" + "-"*40 + "\n")

# Program 11: TF-IDF
print("TF-IDF Matrix:")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([input_text_tfidf])
feature_names = vectorizer.get_feature_names_out()
dense_tfidf = tfidf_matrix.todense()
tfidf_array = np.array(dense_tfidf)

for i in range(tfidf_array.shape[0]):
    print(f"Document {i + 1}:")
    for j in range(tfidf_array.shape[1]):
        if tfidf_array[i][j] > 0:
            print(f" - {feature_names[j]}: {tfidf_array[i][j]}")
print("\n" + "-"*40 + "\n")

# Program 12: Parse Tree Visualization (For Jupyter Notebooks or Streamlit Apps)
# Uncomment the next lines if running in a Jupyter notebook or Streamlit environment
# print("Parse Tree Visualization:")
# doc = nlp("This is a sample sentence for parse tree.")
# for sent in doc.sents:
#     displacy.render(sent, style="dep", jupyter=True)