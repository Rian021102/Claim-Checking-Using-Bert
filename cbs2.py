import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

st.title('Classification Using BERT Apps')


def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('/Users/rianrachmanto/pypro/claimbust/modelsave/')
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter Text To Analyze')
button = st.button('Analyze')
d = {
    1:'Claim',
    0:'Non Claim'
}

if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=500, return_tensors='pt')
    output = model(**test_sample)
    st.write('logits: ', output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    st.write('Prediction:', d[y_pred[0]])



