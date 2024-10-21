# from flask import Flask, render_template, request, send_file, redirect, url_for
# import spacy
# import nltk
# from nltk.stem import PorterStemmer
# from nltk import CFG, ChartParser
# from io import BytesIO
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from reportlab.lib import colors
# from nltk.tree import Tree
# import graphviz
# import os
# from sklearn.feature_extraction.text import CountVectorizer

# # Download required NLTK resources
# nltk.download('punkt')

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Initialize stemmer
# stemmer = PorterStemmer()

# # Define the grammar for parsing

# grammar = CFG.fromstring("""
#     S -> NP VP
#     NP -> PR N
#     NP -> PR N P NP
#     NP -> NP P NP
#     VP -> V NP
#     VP -> V NP P NP
#     VP -> V ADV
#     VP -> V NP ADV
#     VP -> V ADV P NP
#     VP -> V NP ADV P NP

#     PR -> 'a' | 'the' | 'A' | 'The' 
#     N -> 'dog' | 'cat' | 'girl' | 'boy' | 'treat' | 'park'
#     V -> 'chases' | 'sees' | 'finds' | 'gives' | 'takes' | 'runs'
#     ADV -> 'quickly' | 'happily' | 'silently'
#     P -> 'in' | 'to' | 'at' | 'with'
# """)


# PREDEFINED_SENTENCES = [
#     "The quick fox jumps over the lazy dog",
#     "The happy rabbit runs near the small house",
#     "The lazy cat sleeps on the red car",
#     "An old turtle finds a blue mouse",
#     "The quick brown fox chases a green rabbit",
#     "The furry dog likes to sit on the grass",
#     "A small bird flies over the tall building",
#     "The happy dog barks with a loud voice",
#     "The tall rabbit sits under a green tree",
# ]



# parser = ChartParser(grammar)

# app = Flask(__name__)

# # Main route for displaying the home page
# @app.route('/')
# def index():

    
#     return render_template('index.html', result="", parse_tree_image=None)


# ## Route for processing text
# @app.route('/process', methods=['POST'])
# def process():
#     input_text = request.form['input_text']
#     operations = request.form.getlist('operations')

#     if not input_text.strip():
#         return render_template('index.html', error="Please enter some text to process.")

#     result = ""
#     parse_tree_image_path = None

#     # Process the text using spaCy
#     doc = nlp(input_text)

#     # Apply selected operations
#     if 'tokenization' in operations:
#         tokens = [token.text for token in doc]
#         result += "Tokenization:\n" + "\n".join(tokens) + "\n\n"

#     if 'segmentation' in operations:
#         sentences = [sent.text for sent in doc.sents]
#         result += "Segmentation:\n" + "\n".join(sentences) + "\n\n"

#     if 'lowercasing' in operations:
#         lowercased_tokens = [token.text.lower() for token in doc]
#         result += "Lowercasing:\n" + "\n".join(lowercased_tokens) + "\n\n"

#     if 'stopword_removal' in operations:
#         filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
#         result += "Tokens after Stop Word Removal:\n" + "\n".join(filtered_tokens) + "\n\n"

#     if 'remove_digits_punct' in operations:
#         cleaned_tokens = [token.text for token in doc if token.is_alpha]
#         result += "Tokens after Removing Punctuation/Digits:\n" + "\n".join(cleaned_tokens) + "\n\n"

#     if 'stemming' in operations:
#         stemmed_tokens = [stemmer.stem(token.text) for token in doc if token.is_alpha]
#         result += "Stemming:\n" + "\n".join(stemmed_tokens) + "\n\n"

#     if 'lemmatization' in operations:
#         lemmatized_tokens = [token.lemma_ for token in doc]
#         result += "Lemmatization:\n" + "\n".join(lemmatized_tokens) + "\n\n"

#     if 'pos_tagging' in operations:
#         pos_tags = [f"{token.text} ({token.pos_})" for token in doc]
#         result += "POS Tagging:\n" + "\n".join(pos_tags) + "\n\n"

#     if 'bag_of_words' in operations:
#         # Stop word removal and Bag of Words generation
#         filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
#         # Create Bag of Words using the filtered tokens
#         vectorizer = CountVectorizer()
#         X = vectorizer.fit_transform([" ".join(filtered_tokens)])  # Join tokens back into a single string
        
#         # Handling versions of scikit-learn
#         try:
#             words = vectorizer.get_feature_names_out()  # Newer versions
#         except AttributeError:
#             words = vectorizer.get_feature_names()  # For older versions

#         vector = X.toarray().flatten()

#         result += "Bag of Words:\n"
#         for word, count in zip(words, vector):
#             result += f"{word}: {count}\n"

#         result += "\nBag of Words Vector Representation:\n"
#         result += str(vector) + "\n\n"

#     if 'parse_tree' in operations:
#         sentence = input_text.split()
#         try:
#             parse_tree_list = list(parser.parse(sentence))
#             if parse_tree_list:
#                 tree = parse_tree_list[0]

#                 # Generate the Graphviz parse tree image
#                 dot = graphviz.Digraph()

#                 def add_edges(tree):
#                     dot.node(str(tree), str(tree.label()))
#                     for child in tree:
#                         if isinstance(child, Tree):
#                             add_edges(child)
#                             dot.edge(str(tree), str(child))
#                         else:
#                             dot.node(str(child), str(child))
#                             dot.edge(str(tree), str(child))

#                 add_edges(tree)
#                 parse_tree_image_path = 'static/parse_tree'  # Path without extension
#                 dot.render(parse_tree_image_path, format='png', cleanup=True)

#                 result += "Parse Tree generated successfully.\n\n"
#             else:
#                 result += "Parse Error: The input text could not be parsed using the defined grammar.\n\n"
#         except Exception as e:
#             result += f"Parse Error: {e}\n\n"

#     # Return the result properly indented within the process function
#     return render_template('index.html', result=result, parse_tree_image=parse_tree_image_path if 'parse_tree' in operations else None)

# # Route for generating PDF and downloading
# @app.route('/download', methods=['POST'])
# def download_pdf():
#     processed_output = request.form.get('result', '')

#     if not processed_output.strip():
#         return "No content to generate PDF", 400  # Return a 400 error if there's no content

#     buffer = BytesIO()
#     c = canvas.Canvas(buffer, pagesize=letter)
#     width, height = letter

#     c.setFont("Helvetica-Bold", 18)
#     c.setFillColor(colors.darkblue)
#     c.drawString(40, height - 30, "Bitbuilder Productions")

#     lines = processed_output.splitlines()
#     y_position = height - 60
#     line_height = 14
#     margin = 40

#     operations = [
#         "Tokenization",
#         "Segmentation",
#         "Lowercasing",
#         "Tokens after Stop Word Removal",
#         "Tokens after Removing Punctuation/Digits",
#         "Stemming",
#         "Lemmatization",
#         "POS Tagging",
#         "Parse Tree"
#     ]

#     for line in lines:
#         if y_position < line_height + 40:
#             c.showPage()
#             c.setFont("Helvetica-Bold", 18)
#             c.setFillColor(colors.darkblue)
#             c.drawString(40, height - 30, "Bitbuilder Productions")
#             y_position = height - 60

#         for operation in operations:
#             if line.startswith(operation + ":"):
#                 c.setFont("Helvetica-Bold", 16)
#                 c.setFillColor(colors.lightblue)
#                 c.drawString(margin, y_position, line)
#                 y_position -= line_height
#                 break
#         else:
#             c.setFont("Courier", 10)
#             c.setFillColor(colors.black)
#             c.drawString(margin, y_position, line)

#         y_position -= line_height  # Move down for the next line

#     c.save()

#     buffer.seek(0)  # Move to the beginning of the BytesIO buffer
#     return send_file(buffer, as_attachment=True, download_name="processed_output.pdf", mimetype='application/pdf')

# @app.route('/download_text', methods=['POST'])
# def download_text():
#     sentences = PREDEFINED_SENTENCES
#     text_file_path = 'sentences.txt'

#     with open(text_file_path, 'w') as f:
#         for sentence in sentences:
#             f.write(f"{sentence}\n")

#     return send_file(text_file_path, as_attachment=True)

# # Route for downloading parse tree image
# @app.route('/download_parse_tree', methods=['GET'])
# def download_parse_tree():
#     parse_tree_image_path = request.args.get('path')
#     if parse_tree_image_path and os.path.exists(parse_tree_image_path):
#         return send_file(parse_tree_image_path, as_attachment=True)
#     return "Parse tree image not found.", 404

# if __name__ == '__main__':
#     os.makedirs('static', exist_ok=True)  # Ensure the static directory exists
#     app.run(debug=True)



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
