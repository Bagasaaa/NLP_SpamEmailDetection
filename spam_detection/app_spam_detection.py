from flask import Flask, flash, request, redirect, url_for, render_template, Markup, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.models import load_model


app = Flask(__name__, template_folder='templates')
app.secret_key = 'bagas_data_science'

def cleansing(text):
    # Make sentence being lowercase
    text = text.lower()
    
    # Decode bytes to string
    text = text.decode('latin-1')

    # Remove hashtag
    pattern_3 = r'#([^\s]+)'
    text = re.sub(pattern_3, '', text)

    # Remove general punctuation, math operation char, etc.
    pattern_4 = r'[\,\@\*\_\-\!\:\;\?\'\.\"\)\(\{\}\<\>\+\%\$\^\#\/\`\~\|\&\|]'
    text = re.sub(pattern_4, ' ', text)

    # Remove emoji
    pattern_6 = r'\\[a-z0-9]{1,5}'
    text = re.sub(pattern_6, '', text)

    # Remove (\); ([); (])
    pattern_9 = r'[\\\]\[]'
    text = re.sub(pattern_9, '', text)

    # Remove character non ASCII
    pattern_10 = r'[^\x00-\x7f]'
    text = re.sub(pattern_10, '', text)

    # Remove character non ASCII
    pattern_11 = r'(\\u[0-9A-Fa-f]+)'
    text = re.sub(pattern_11, '', text)

    # Remove multiple whitespace
    pattern_12 = r'(\s+|\\n)'
    text = re.sub(pattern_12, ' ', text)
    
    # Remove whitespace at the first and end sentences
    text = text.rstrip()
    text = text.lstrip()
    return text

def tokenisasi(text):
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

##### home interface as .html
@app.route("/", methods=['GET'])
def home():
    return render_template('home_spam.html')

@app.route("/spam_or_ham", methods=['GET', 'POST'])
def clean():
    if request.method == 'POST':
        tweet = request.form['tweet']
        result = cleansing(tweet)
        result = [result]
        result = pd.DataFrame({'text' : result})
        result = pd.DataFrame(result['text'])

        sentences = result['text'].to_list()

        MAX_NB_WORDS = 50000
        MAX_SEQUENCE_LENGTH = 250
        EMBEDDING_DIM = 100
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(sentences)
        word_index = tokenizer.word_index
        X_new = tokenizer.texts_to_sequences(sentences)
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("email_spam_classifier.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "spam", 1: "ham"}
        result = [labels[pred] for pred in y_pred]
        result = result[0]
            
        return redirect(url_for("cleansing", text=result))

    return render_template("input_text_spam.html")

@app.route("/<text>", methods=['GET'])
def cleansing(text):
    return f'Classification: {text}'

if __name__ == '__main__':
    app.run(debug=True)