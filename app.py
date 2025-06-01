import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

import asyncio
import nest_asyncio  
nest_asyncio.apply()

import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def fix_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
fix_event_loop()

@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)  
    return tokenizer, model, device  

tokenizer, model, device = load_model() 

st.set_page_config(page_title="Summarizer", page_icon="📝")
st.title("📝 Text Summarizer")

text = st.text_area("Enter text to summarize", height=200)
max_length = st.slider("Max summary length", 30, 300, 130)
min_length = st.slider("Min summary length", 10, 100, 30)

def summarize(text):
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if st.button("Summarize"):
    if text.strip():
        with st.spinner("Summarizing..."):
            result = summarize(text)
        st.success(result)
    else:
        st.warning("Please enter some text.")
