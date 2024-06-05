import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import re
import time  # Add this line to import the time module

class SentimentAnalysisRestaurant:
    def __init__(self):
        st.title("Sentiment Analysis Restaurant")
        # Add image below the title
        st.image("images.jpeg", caption="Hello There and Welcome", use_column_width=True)
        self.entry = st.text_input("Type your review here....", key="entry")
        self.prediction_result = st.empty()

    def predict(self):
        start_time = time.time()

        def lowercase(text):
            return text.lower()

        def remove_unnecessary_char(text):
            text = re.sub('\n', ' ', text)
            text = re.sub('rt', ' ', text)
            text = re.sub('user', ' ', text)
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
            text = re.sub('  +', ' ', text)
            text = re.sub('ï¿½', ' ', text)
            text = re.sub('ï¿', ' ', text)
            text = re.sub('ý', ' ', text)
            text = re.sub('ï', ' ', text)
            return text

        def remove_symbolnumeric(text):
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text

        def preprocess(text):
            text = lowercase(text)
            text = remove_unnecessary_char(text)
            text = remove_symbolnumeric(text)
            return text

        with open('vokabuler.pkl', 'rb') as file:
            vokabuler_loaded = pickle.load(file)

        vectorizer_baru = TfidfVectorizer(vocabulary=vokabuler_loaded)

        teks_baru = self.entry
        teks_baru = preprocess(teks_baru)
        teks_baru_transformed = vectorizer_baru.fit_transform([teks_baru])

        svm_model = joblib.load('svm_tfidf.sav')
        prediksi = svm_model.predict(teks_baru_transformed)

        end_time = time.time()
        running_time = end_time - start_time

        emotions_emoji_dict = {
            "Very Negative": "😠",
            "Very Positive": "😁",
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😔",
        }

        # Update the Streamlit app with the prediction result and running time
        sentiment_label = "Sentiment: {}".format(prediksi[0])
        emoji = emotions_emoji_dict.get(prediksi[0], "")
        self.prediction_result.text(sentiment_label + " " + emoji)
        st.text("Running Time: {:.4f} seconds".format(running_time))
        print("Sentiment:", prediksi[0])

# Instantiate the app
app = SentimentAnalysisRestaurant()

# Add a button to trigger prediction
if st.button("Predict"):
    app.predict()
