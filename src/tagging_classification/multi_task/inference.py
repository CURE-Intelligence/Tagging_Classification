from transformers import  AutoTokenizer
from checkpoint import load_model
import torch
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()


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

if __name__ == '__main__':

    model_dir = os.getenv('MODEL_DIR')

    #print(model_dir)

    tokenizer_name = "gerticure/tokenizer_v1"

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

    model = load_model(model_dir, "pytorch_model_v2.bin")  # Loaded on the cpu
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text = [
        "900 Milliarden US-Dollar DWS führt physischen Bitcoin-ETC ein In Deutschland hat die DWS  ein bekannter Vermögensverwalter mit einem verwalteten Vermögen von über 900 Milliarden US-Dollar in Zusammenarbeit mit Galaxy Digital Holdings Ltd. neue Xtrackers Exchange-Traded Commodities (ETC) eingeführt  die Anlegern einen bequemen Zugang zu Bitcoin-Engagements bieten"]

    #text = ["Classic Optionsscheine auf DWS Group GmbH | DWS100, nan"]



    outputs = predict(text, model, tokenizer, article_type_labels = article_type_labels, categorie_labels =  categorie_labels)

    print(outputs[0]['predicted_article_type'])
    print(outputs[0]['predicted_categorie'])