import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from collections import Counter
import random
from tqdm.auto import tqdm
import os


def convert_to_hf_data(df):
    # First, split off the test set (5% of the data)
    train_val_df, test_df = train_test_split(df, test_size=0.05, random_state=42)

    # Then split the remaining data into train and validation
    # 0.15 / (0.8 + 0.15) ≈ 0.1579 gives us the correct proportion for validation
    train_df, val_df = train_test_split(train_val_df, test_size=0.1579, random_state=42)

    # Convert data to huggingface datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset

def rename_target_variable(hf_dataset):

    #Rename the target variable from encoded_tags to labels
    return hf_dataset.rename_columns({'article_type_categorie' : 'labels'})

def remove_columns(hf_dataset, columns):
    return hf_dataset.remove_columns(columns)

def set_format_to_pytorch(hf_dataset):
    # Set the format of the data inside the variable as torch
    hf_dataset.set_format("torch")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "f1_score": f1_score(labels, predictions, average='weighted'),
        "accuracy": accuracy_score(labels, predictions)
    }



