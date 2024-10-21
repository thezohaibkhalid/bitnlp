from sklearn.feature_extraction.text import CountVectorizer
import spacy


nlp = spacy.load("en_core_web_sm")

def generate_bag_of_words(text):

    doc = nlp(text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])  

    words = vectorizer.get_feature_names_out()
    vector = X.toarray().flatten()

    result = "Bag of Words:\n"
    for word, count in zip(words, vector):
        result += f"{word}: {count}\n"

    print(result)

# Predefined paragraph
input_text = """This is a sample text for generating a bag of words. 
It includes stop words like the, a, and other common words to demonstrate the functionality."""

generate_bag_of_words(input_text)
