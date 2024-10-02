from transformers import  AutoTokenizer
from src.tagging_classification.multi_task.checkpoint import load_model
import torch
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st


@st.cache_resource
def load_model_and_tokenizer():
    model_dir = os.getenv('MODEL_DIR')
    tokenizer_name = "gerticure/tokenizer_v1"
    model = load_model(model_dir, "pytorch_model_v3.bin")  # Loaded on the cpu
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def predict(texts, model, tokenizer, **kwargs):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    results = []

    # Perform inference
    model.eval()  # Ensure the model is in evaluation mode

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the outputs
    article_type_preds = torch.argmax(outputs['article_type_logits'], dim=-1)
    categorie_preds = torch.argmax(outputs['category_logits'], dim=-1)

    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'predicted_article_type': kwargs['article_type_labels'][article_type_preds[i]],
            'predicted_categorie': kwargs['categorie_labels'][categorie_preds[i]]
        })
    # return results
    return results

#Streamlit UI
st.title("Tagging Classification App")

#Load Model and Tokenizer
model, tokenizer = load_model_and_tokenizer()

# Define labels
article_type_labels = [
    "is clearly written by a DWS employee",
    "no information given"
]

categorie_labels = [
    "contains information about illiquid assets like real estates, loans, private equity, or other long-term investments",
    "contains information about liquid assets like ETF bonds, treasury bills, or funds",
    "describes corporate news about DWS",
    "DWS acts as an expert",
    "no information given"
]

#Define text input
user_input = st.text_area("Enter the text you want to classify", height=200)

if st.button("Classify"):
    if user_input:
        # Make prediction
        results = predict([user_input], model, tokenizer,
                          article_type_labels=article_type_labels,
                          categorie_labels=categorie_labels)

        # Display results
        st.subheader("Classification Results:")
        st.text(f"Article Type : {results[0]['predicted_article_type']}")
        st.text(f"Categorie : {results[0]['predicted_categorie']}")
    else:
        st.warning("Please enter some text to classify.")