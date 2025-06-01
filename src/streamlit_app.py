import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("📝 Text Summarizer")

text = st.text_area("Enter text to summarize", height=200)
max_length = st.slider("Max summary length", 30, 300, 130)

if st.button("Summarize"):
    if text.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(text, max_length=max_length, min_length=30)[0]['summary_text']
        st.success(summary)
    else:
        st.warning("Please enter some text.")
