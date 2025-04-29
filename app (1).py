
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ml_model = pickle.load(open("ml_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
dl_model = tf.keras.models.load_model("dl_model.h5")
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

st.set_page_config(page_title="📊 Financial Sentiment Analyzer", layout="centered")
st.markdown("<h1 style='color:#4CAF50;'>📈 Financial News Sentiment Analyzer</h1>", unsafe_allow_html=True)

text = st.text_area("Enter Financial Headline:")

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        tfidf_input = tfidf.transform([text])
        ml_pred = ml_model.predict_proba(tfidf_input)[0]
        ml_label = np.argmax(ml_pred)

        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=20)
        dl_pred = dl_model.predict(pad)[0]
        dl_label = np.argmax(dl_pred)

        st.subheader("Results")
        st.markdown(f"🧠 **ML Model:** {label_map[ml_label]} ({ml_pred[ml_label]*100:.2f}%)")
        st.markdown(f"🔮 **DL Model:** {label_map[dl_label]} ({dl_pred[dl_label]*100:.2f}%)")
