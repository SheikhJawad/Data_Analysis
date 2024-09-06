import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = 'jerry124/finetuned_spam_ham_classifier'
standard_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_name)

def classify_text(text):
 
    inputs = standard_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
  
    with torch.no_grad():
        outputs = model(**inputs)
    
  
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    return "Spam" if predicted_class == 1 else "Ham"

st.title("Spam/Ham Classifier")
st.write("Enter text to classify whether it is spam or ham.")

user_input = st.text_area("Input Text")

if st.button("Classify"):
    if user_input:
        result = classify_text(user_input)
        st.write(f"The text is classified as: **{result}**")
    else:
        st.write("Please enter some text to classify.")
