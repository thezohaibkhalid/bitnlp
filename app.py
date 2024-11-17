from flask import Flask, render_template, request, send_file
import spacy
from benepar import BeneparComponent, download
from nltk import Tree
from spacy.tokens import Doc
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import os
from sklearn.feature_extraction.text import CountVectorizer
import graphviz

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Add Benepar to the pipeline
download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# Register the extension attribute on the Doc object
Doc.set_extension("parse_string", default=None)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result="", parse_tree_image=None)

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    operations = request.form.getlist('operations')
    if not input_text.strip():
        return render_template('index.html', error="Please enter some text to process.")

    result = ""
    parse_tree_image_path = None

    doc = nlp(input_text)

    if 'tokenization' in operations:
        tokens = [token.text for token in doc]
        result += "Tokenization:\n" + "\n".join(tokens) + "\n\n"

    if 'segmentation' in operations:
        sentences = [sent.text for sent in doc.sents]
        result += "Segmentation:\n" + "\n".join(sentences) + "\n\n"

    if 'lowercasing' in operations:
        lowercased_tokens = [token.text.lower() for token in doc]
        result += "Lowercasing:\n" + "\n".join(lowercased_tokens) + "\n\n"

    if 'stopword_removal' in operations:
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        result += "Tokens after Stop Word Removal:\n" + "\n".join(filtered_tokens) + "\n\n"

    if 'remove_digits_punct' in operations:
        cleaned_tokens = [token.text for token in doc if token.is_alpha]
        result += "Tokens after Removing Punctuation/Digits:\n" + "\n".join(cleaned_tokens) + "\n\n"

    if 'stemming' in operations:
        stemmed_tokens = [token.lemma_ for token in doc if token.is_alpha]
        result += "Stemming:\n" + "\n".join(stemmed_tokens) + "\n\n"

    if 'lemmatization' in operations:
        lemmatized_tokens = [token.lemma_ for token in doc]
        result += "Lemmatization:\n" + "\n".join(lemmatized_tokens) + "\n\n"

    if 'pos_tagging' in operations:
        pos_tags = [f"{token.text} ({token.pos_})" for token in doc]
        result += "POS Tagging:\n" + "\n".join(pos_tags) + "\n\n"

    if 'bag_of_words' in operations:
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([" ".join(filtered_tokens)])
        words = vectorizer.get_feature_names_out()
        vector = X.toarray().flatten()
        result += "Bag of Words:\n"
        for word, count in zip(words, vector):
            result += f"{word}: {count}\n"
        result += "\nBag of Words Vector Representation:\n"
        result += str(vector) + "\n\n"

        # Ensure the word embedding operation is correctly indented
    if 'word_embedding' in operations:
        # Split input_text into documents (each sentence as a separate document)
        documents = input_text.strip().split('\n')  # Each line as a separate document

        # Create a CountVectorizer to generate word embeddings
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)  # Transform documents into word vectors

        # Get the feature names (words) and their corresponding vectors
        feature_names = vectorizer.get_feature_names_out()

        # Example: Get embedding for a specific word (e.g., 'Lahore')
        word = 'Lahore'
        if word in feature_names:
            word_index = np.where(feature_names == word)[0][0]  # Find the index of the word
            word_vector = X[:, word_index].toarray().flatten()  # Extract the embedding vector
            result += f"Word Embedding for '{word}': {word_vector}\n\n"
        else:
            result += f"'{word}' not in vocabulary.\n\n"

    
    if 'tfidf' in operations:
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform([input_text])  # Ensure input_text is the document
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Convert the TF-IDF matrix to a dense format and then to an array
        dense_tfidf = tfidf_matrix.todense()
        tfidf_array = np.array(dense_tfidf)

        # Collect TF-IDF results
        result += "TF-IDF Matrix:\n\n"
        for i in range(tfidf_array.shape[0]):
            result += f"Document {i + 1}:\n"
            for j in range(tfidf_array.shape[1]):
                if tfidf_array[i][j] > 0:  # Print only non-zero TF-IDF values
                    result += f" - {feature_names[j]}: {tfidf_array[i][j]}\n"
            result += "\n"

    if 'parse_tree' in operations:
        for sent in doc.sents:
            parse_string = sent._.parse_string
            if parse_string:
                tree = Tree.fromstring(parse_string)
                dot = graphviz.Digraph()
                
                def add_edges(tree):
                    dot.node(str(tree), str(tree.label()))
                    for child in tree:
                        if isinstance(child, Tree):
                            add_edges(child)
                            dot.edge(str(tree), str(child))
                        else:
                            dot.node(str(child), str(child))
                            dot.edge(str(tree), str(child))
                
                add_edges(tree)
                parse_tree_image_path = 'static/parse_tree'
                dot.render(parse_tree_image_path, format='png', cleanup=True)
                result += "Parse Tree generated successfully.\n\n"

    return render_template('index.html', result=result, parse_tree_image=parse_tree_image_path if 'parse_tree' in operations else None)

@app.route('/download', methods=['POST'])
def download_pdf():
    processed_output = request.form.get('result', '')
    if not processed_output.strip():
        return "No content to generate PDF", 400
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawString(40, height - 30, "Bitbuilder Productions")
    lines = processed_output.splitlines()
    y_position = height - 60
    line_height = 14
    margin = 40
    operations = [
        "Tokenization",
        "Segmentation",
        "Lowercasing",
        "Tokens after Stop Word Removal",
        "Tokens after Removing Punctuation/Digits",
        "Stemming",
        "Lemmatization",
        "POS Tagging",
        "Parse Tree"
    ]
    for line in lines:
        if y_position < line_height + 40:
            c.showPage()
            c.setFont("Helvetica-Bold", 18)
            c.setFillColor(colors.darkblue)
            c.drawString(40, height - 30, "Bitbuilder Productions")
            y_position = height - 60
        for operation in operations:
            if line.startswith(operation + ":"):
                c.setFont("Helvetica-Bold", 16)
                c.setFillColor(colors.lightblue)
                c.drawString(margin, y_position, line)
                y_position -= line_height
                break
        else:
            c.setFont("Courier", 10)
            c.setFillColor(colors.black)
            c.drawString(margin, y_position, line)
        y_position -= line_height
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="processed_output.pdf", mimetype='application/pdf')

@app.route('/download_text', methods=['POST'])
def download_text():
    sentences = request.form.get('sentences', '')
    text_file_path = 'sentences.txt'
    with open(text_file_path, 'w') as f:
        f.write(sentences)
    return send_file(text_file_path, as_attachment=True)

@app.route('/download_parse_tree', methods=['GET'])
def download_parse_tree():
    parse_tree_image_path = request.args.get('path')
    if parse_tree_image_path and os.path.exists(parse_tree_image_path):
        return send_file(parse_tree_image_path, as_attachment=True)
    return "Parse tree image not found.", 404

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
//app changes