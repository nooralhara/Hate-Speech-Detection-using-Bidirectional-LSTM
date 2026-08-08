import streamlit as st
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Must match training: max_length and vocab_size used when the tokenizer
# and model were trained in the notebook.
max_length = 100

model = load_model('hate_speech_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class_names = ['Hate Speech', 'Offensive Language', 'Neither']


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


st.set_page_config(page_title='Hate Speech Detection', page_icon='🛡️')

st.title('Hate Speech Detection')
st.write('Enter text below to check whether it contains hate speech.')

user_input = st.text_area('Text input', height=150)

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text.')
    else:
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        prediction = model.predict(padded)
        predicted_class = prediction.argmax(axis=1)[0]
        confidence = prediction[0][predicted_class]

        st.subheader('Prediction: ' + class_names[predicted_class])
        st.write('Confidence: ' + str(round(confidence * 100, 2)) + '%')

        with st.expander('Show class probabilities'):
            index = 0
            while index < len(class_names):
                st.write(class_names[index] + ': ' + str(round(prediction[0][index] * 100, 2)) + '%')
                index = index + 1
