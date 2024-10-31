from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

app = Flask(__name__)
CORS(app)

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


class Preprocess:
    @staticmethod
    def remove_punctuations(text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif isinstance(text, np.ndarray):
            text = text.astype(str)
        removed_punctuations = re.sub(r'[^\w\s]', '', text)
        return removed_punctuations

    @staticmethod
    def tokenizer(text):
        return word_tokenize(text)

    @staticmethod
    def remove_stop_words(tokenized_text):
        stop_words = set(stopwords.words('english'))
        words_filtered = [
            word for word in tokenized_text if word.isalpha() and word not in stop_words]
        return words_filtered

    @staticmethod
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                    "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def lemmatize(tokenized_text):
        lemmatizer = WordNetLemmatizer()
        words_filtered = [lemmatizer.lemmatize(
            word, Preprocess.get_wordnet_pos(word)) for word in tokenized_text]
        return words_filtered

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = Preprocess.remove_punctuations(text)
        tokenized_text = Preprocess.tokenizer(text)
        tokenized_text = Preprocess.remove_stop_words(tokenized_text)
        lemmatized_text = Preprocess.lemmatize(tokenized_text)
        return ' '.join(lemmatized_text)


label_mapping = {0: 'JAUNDICE', 1: 'PERINATAL ASPHYXIA',
                 2: 'PREMATURITY', 3: 'SEPSIS'}


@app.route('/')
def home():
    return render_template('web_app.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms_test = data['content']
    preprocessed_input = Preprocess.preprocess_text(symptoms_test)
    print(f'Preprocessed_Text:{preprocessed_input}')
    model = tf.keras.models.load_model('models/ANN_Model.keras')
    prediction = model.predict(np.array([preprocessed_input]).astype(object))
    print(f'Prediction:{prediction}')
    pred_label = label_mapping[np.argmax(prediction)]
    print(pred_label)
    return jsonify({'prediction': pred_label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
