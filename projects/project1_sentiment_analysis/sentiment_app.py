import streamlit as st
from transformers import pipeline

# Load the Hugging Face pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_pipeline()

# Streamlit app layout
st.title("Sentiment Analysis App")
st.write("Enter a sentence or phrase to determine its sentiment.")

# Input text from user
user_input = st.text_area("Enter text here:")

# Analyze sentiment when the user provides input
if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
    else:
        st.warning("Please enter some text for analysis.")
