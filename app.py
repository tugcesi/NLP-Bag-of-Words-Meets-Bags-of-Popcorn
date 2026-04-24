import streamlit as st
import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow import keras
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Film Yorumu Duygu Analizi",
    page_icon="🎬",
    layout="centered",
)

# ─── Model & Vectorizer Yükleme ───────────────────────────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    model = keras.models.load_model("bagofwords_model.h5")
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ─── Metin Ön İşleme ──────────────────────────────────────────────────────────
def preprocess_text(raw_text: str) -> str:
    text = BeautifulSoup(raw_text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.lower().split()
    stop_words = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stop_words]
    return " ".join(meaningful_words)

# ─── Tahmin Fonksiyonu ────────────────────────────────────────────────────────
def predict_sentiment(text: str):
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    prob = model.predict(features, verbose=0)[0][0]
    label = "😊 Pozitif" if prob >= 0.5 else "😞 Negatif"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(confidence), float(prob)

# ─── Arayüz ───────────────────────────────────────────────────────────────────
st.title("🎬 Film Yorumu Duygu Analizi")
st.markdown(
    "Bir film yorumu gir; model bunun **pozitif mi** yoksa **negatif mi** "
    "olduğunu tahmin etsin."
)
st.divider()

user_input = st.text_area(
    "Film yorumunu buraya yaz:",
    placeholder="Örnek: This movie was absolutely fantastic! The acting was superb...",
    height=160,
)

if st.button("🔍 Analiz Et", use_container_width=True):
    if not user_input.strip():
        st.warning("Lütfen bir yorum girin.")
    else:
        with st.spinner("Tahmin yapılıyor..."):
            label, confidence, raw_prob = predict_sentiment(user_input)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Tahmin", label)
        with col2:
            st.metric("Güven Skoru", f"%{confidence * 100:.1f}")

        st.progress(raw_prob, text=f"Pozitif olasılığı: {raw_prob:.3f}")

        with st.expander("🔎 Ön işlenmiş metin"):
            st.code(preprocess_text(user_input))

st.divider()
st.caption("Bag of Words · Keras · IMDB Movie Reviews")
