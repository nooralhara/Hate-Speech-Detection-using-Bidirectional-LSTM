import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

st.title("Hate Speech Detection")
st.write("Enter a tweet or sentence to classify it.")

@st.cache_resource
def load_assets():
    model = load_model('bilstm_hate_speech.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

max_length = 15

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+\www\S+", "", text)
    text = re.sub(r"<.&?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

labels = {
    0: "Hate Speech",
    1: "Not Hateful"
}

user_input = st.text_area("Your text:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        st.subheader(f"Prediction: {labels[predicted_class]}")
        st.write(f"Confidence: {confidence:.2%}")

        st.write("Class probabilities:")
        for i, label in labels.items():
            st.write(f"{label}: {prediction[0][i]:.2%}")
