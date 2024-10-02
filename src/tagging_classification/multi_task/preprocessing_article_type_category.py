import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import random
from tqdm.auto import tqdm
import os



#remove dots at the end of the sentence
def remove_dots(text):

    '''Remove dots at the end of the sentence'''
    return text.rstrip('.')


def replace_pattern_with_right_text(text):

    '''
        Replace several patterns where:
        -no/No information given appears
        -DWS employee appears
        -describes corporate news about DWS appears
        -DWS acts as an expert appears
        -'liquid' word appears (and other variations of liquid based on the language is used)
        -'illiquid' word appears (and other variations of illiquid based on the language is used)

        Replace them with:
        -no information given
        -is clearly written by a DWS employee
        -describes corporate news about DWS
        -if DWS acts as an expert
        -contains information about liquid assets
        -contains information about illiquid assets
    '''

    no_info_pattern = r'\b([nN]o\s+inform?a?t?i?o?n?|[nN]o\s+info|[kK]eine\s+[iI]nformationen|[nN]essuna\s+informazione|[nN]o\s+información\s+dada)\b'
    dws_employee_pattern = r'\bDWS\s+employee\b'
    dws_news_pattern = r'\bdescribes corporate news about DWS\b'
    dws_act_as_expert = r'\b(DWS acts as an expert|DWS\s+expert|expert)\b'
    liquid_pattern = r'\b(ETFs?|ETFs?\s+(liquid|líquidos|liquid assets|liquide)|(?:liquid|líquidos)\s+ETFs?|activos\s+líquidos)\b'
    illiquid_pattern = r'\billiquid|ilíquidos\b'


    if pd.notna(text):
        if pd.Series(text).str.contains(no_info_pattern, regex=True).any():
            return 'no information given'
        elif pd.Series(text).str.contains(dws_employee_pattern, regex=True).any():
            return 'is clearly written by a DWS employee'
        elif pd.Series(text).str.contains(dws_news_pattern, regex=True).any():
            return 'describes corporate news about DWS'
        elif pd.Series(text).str.contains(dws_act_as_expert, regex=True).any():
            return 'if DWS acts as an expert'
        elif pd.Series(text).str.contains(liquid_pattern, regex=True).any():
            return 'contains information about liquid assets like ETFs, bonds, treasury bills, or funds'
        elif pd.Series(text).str.contains(illiquid_pattern, regex=True).any():
            return 'contains information about illiquid assets, like real estates, loans private equity, or other long term-investments'

        return 'contains information about liquid assets like ETFs, bonds, treasury bills, or funds'




############################Preprocessing the data using Huggingface datasets#############################
def convert_to_hf_data(df):

    """
    This function will perform train test splitting on our original dataset will convert then the obtained dataframes into huggingface datasets
    """

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert data to huggingface datasets
    train_dataset = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])
    val_dataset = Dataset.from_pandas(val_df).remove_columns(['__index_level_0__'])

    return train_dataset, val_dataset

def str_to_int(s):
    try:
        return int(s)
    except ValueError:
        print(f"Error: Cannot convert '{s}' to an integer.")
        return None

def tokenize_dataset(example, tokenizer):
    tokenized_batch =  tokenizer(example['Volltext'], truncation=True, padding="max_length", max_length=512 )

    return tokenized_batch




def format_labels(example):
    return {
        'labels': {
            'article_type': example['article_type'],
            'categorie': example['categorie']
        }
    }


def compute_metrics(eval_pred):

    """Computing the metrics"""

    logits, labels = eval_pred
    article_type_logits, category_logits = logits
    article_type_labels = labels['article_type']
    category_labels = labels['categorie']

    article_type_preds = np.argmax(article_type_logits, axis=-1)
    category_preds = np.argmax(category_logits, axis=-1)

    article_type_accuracy = accuracy_score(article_type_labels, article_type_preds)
    category_accuracy = accuracy_score(category_labels, category_preds)

    article_type_f1 = f1_score(article_type_labels, article_type_preds, average='weighted')
    category_f1 = f1_score(category_labels, category_preds, average='weighted')

    return {
        'article_type_accuracy': article_type_accuracy,
        'category_accuracy': category_accuracy,
        'article_type_f1': article_type_f1,
        'category_f1': category_f1,
        'mean_accuracy': (article_type_accuracy + category_accuracy) / 2,
        'mean_f1': (article_type_f1 + category_f1) / 2
    }