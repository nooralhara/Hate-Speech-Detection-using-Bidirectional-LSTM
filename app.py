import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Config
# ---------------------------
MAX_LENGTH = 15
MODEL_PATH = "bilstm_hate_speech.keras"
TOKENIZER_PATH = "tokenizer.pkl"

CLASS_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

CLASS_COLORS = {
    0: "🔴",
    1: "🟠",
    2: "🟢"
}

# ---------------------------
# Same cleaning function used in training
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------
# Load model + tokenizer (cached so it only loads once)
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ---------------------------
# Prediction function
# ---------------------------
def predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    return pred_class, probs

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Hate Speech Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Hate Speech Detection")
st.write("Enter a tweet or sentence below to classify it as **Hate Speech**, **Offensive Language**, or **Neither**.")

user_input = st.text_area("Enter text:", height=120, placeholder="Type or paste text here...")

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        pred_class, probs = predict(user_input)
        label = CLASS_NAMES[pred_class]
        emoji = CLASS_COLORS[pred_class]

        st.subheader(f"{emoji} Prediction: {label}")
        st.write(f"Confidence: **{probs[pred_class]*100:.2f}%**")

        st.write("### Class Probabilities")
        for cls_idx, cls_name in CLASS_NAMES.items():
            st.progress(float(probs[cls_idx]), text=f"{cls_name}: {probs[cls_idx]*100:.2f}%")

st.markdown("---")
st.caption("Model: Bidirectional LSTM trained on the Davidson Hate Speech dataset.")
