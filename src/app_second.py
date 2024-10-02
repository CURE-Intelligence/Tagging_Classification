import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Class names for predictions
classes = ["is clearly written by a DWS employee/no information given",
           "is clearly written by a DWS employee/ contains information about illiquid assets , like real estates, loans, private equity, or other long term investments",
           "is clearly written by a DWS employee/ contains information about liquid assets like ETFs, bonds, treasury bills or funds",
           "is clearly written by a DWS employee/describe corporate news about DWS",
           "is clearly written by a DWS employee/DWS acts as an expert",
           "no information given/contains information about illiquid assets, like real estates, loans, private equity, or other long term investments",
           "no information given/contains information about liquid assets like ETFs, bonds, treasury bills or funds",
           "no information given/describes corporate news about DWS",
           "no information given/DWS acts as an expert",
           "no information given/no information given"]

# Initialize the model and tokenizer
def initialize_model():
    model_name = "gerticure/tagging_classification_modelv1"
    tokenizer_name = "gerticure/tokenizer_single_v1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Perform inference on the data
def perform_inference(data):
    tokenizer, model = initialize_model()
    text_list = list(data['content'].astype(str).values)  # Extract content for inference

    model.eval()
    # Tokenize the text content
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)

    # Perform inference without gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and predicted classes
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=-1).numpy()

    # Convert predicted class indices to class names
    str_classes = [classes[i] for i in predicted_classes]

    # Add the predictions to the DataFrame
    data['predicted_classes'] = str_classes

    return data

# Generate download link for the DataFrame
def generate_download_link(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
    href = f'<a href="data:file/csv;base64,{b64}" download="annotated_data.csv">Download CSV File</a>'
    return href

# Streamlit app function
def app():
    st.title("Inference Prediction on Uploaded Data")

    # Upload button to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Initialize session state to store the DataFrame
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Process the uploaded CSV file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding='latin1')
        st.write("Uploaded DataFrame:")
        st.write(st.session_state.df.head())

    # Button to run inference
    if st.session_state.df is not None and st.button("Run Inference"):
        st.session_state.df = perform_inference(st.session_state.df)
        st.write("Annotated DataFrame with Predictions:")
        st.write(st.session_state.df.head())

    # Button to download the annotated DataFrame
    if st.session_state.df is not None and st.button("Download Annotated DataFrame"):
        st.markdown(generate_download_link(st.session_state.df), unsafe_allow_html=True)

if __name__ == "__main__":
    app()
