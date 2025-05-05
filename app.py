# app.py
import streamlit as st
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load models
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    theme_model = joblib.load('bert_classifier.pkl')
    urgency_model = joblib.load('urgency_model.pkl')
    return tokenizer, bert_model, theme_model, urgency_model

tokenizer, bert_model, theme_model, urgency_model = load_models()

# App UI
st.title("Mental Health Journal Analyzer")
journal_entry = st.text_area("Enter your journal entry:")

if st.button("Analyze"):
    inputs = tokenizer(journal_entry, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    theme = theme_model.predict(embedding)[0]
    urgency = urgency_model.predict(embedding)[0]
    
    st.success(f"**Detected Theme:** {theme.upper()}")
    st.warning(f"**Urgency Level:** {urgency.upper()}")