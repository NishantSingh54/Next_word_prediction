import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max sequence length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Next Word Prediction using LSTM")
st.write("Enter a sentence and predict the next word.")

text = st.text_input(
    "Enter Text",
    placeholder="Machine learning is..."
)

def predict_next_word(text):
    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    # Predict
    predicted = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(predicted)

    # Find corresponding word
    predicted_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break

    return predicted_word

if st.button("Predict Next Word"):
    if text.strip():

        try:
            next_word = predict_next_word(text)

            st.success(
                f"Predicted Next Word: **{next_word}**"
            )

            st.markdown(
                f"### Complete Sentence\n{text} **{next_word}**"
            )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter some text.")