from typing import cast

import streamlit as st
import re
from keras import Model

from keras.src.datasets import imdb
from keras.src.saving import load_model
from keras.src.utils import pad_sequences

# ======================================================================================================================
# Constants
# ======================================================================================================================
OOV_CHAR = 2
INDEX_FROM = 3
MAX_SENT_SIZE = 500
VOCAB_SIZE = 10000

FILEPATH_MODEL = 'model.keras'

# ======================================================================================================================
# Functions
# ======================================================================================================================
data = imdb.load_data(num_words=VOCAB_SIZE)
word_index: dict = imdb.get_word_index()
reversed_word_index: dict = {v: k for k, v in imdb.get_word_index().items()}


def __encode_sentence(words: list[str]) -> list[list[int]]:
    sentence_encoded = [1]  # 1 indicates the start of a sequence

    for word in words:
        if word in word_index and word_index[word] <= VOCAB_SIZE:  # <= as word index starts off at 1
            sentence_encoded.append(word_index[word] + INDEX_FROM)
        else:
            sentence_encoded.append(OOV_CHAR)

    return [sentence_encoded]


def __remove_punctuation(string: str) -> str:
    stripped_string = re.sub(pattern=r'[^\w\s]', repl='', string=string)

    return stripped_string


def __pre_process(sentence: str) -> list[list[int]]:
    stripped_sentence = __remove_punctuation(sentence)
    words = stripped_sentence.lower().split()
    sentence_encoded = __encode_sentence(words)
    sentence_encoded = pad_sequences(sentence_encoded, maxlen=MAX_SENT_SIZE)

    return sentence_encoded


def __predict_likelihood(model: Model, sentence: str) -> float:
    processed_sentence = __pre_process(sentence)
    likelihood = model.predict(processed_sentence)[0][0]

    return likelihood


# ======================================================================================================================
# Web-app
# ======================================================================================================================
st.set_page_config(page_title="Simple Sentiment RNN")
st.title("Simple Sentiment RNN")
st.caption("Basic neural network trained to analyse whether a movie review is positive or negative.")

review = st.text_input(label="Review", value="This is a sample amazing review")

# ======================================================================================================================
# Predict
# ======================================================================================================================
model = cast(Model, load_model(filepath=FILEPATH_MODEL))
likelihood = __predict_likelihood(model, review)

# ======================================================================================================================
# Writing predictions to website
# ======================================================================================================================
st.divider()
st.write(f"**Likelihood**: {likelihood}")
st.write(f"The statement is **{"positive" if likelihood > 0.5 else "negative"}**.")
