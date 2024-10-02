import pandas as pd
from typing import List, Union
import os
from dotenv import load_dotenv
import numpy as np
import re
import gensim
from gensim.utils import simple_preprocess
from typing import List, Generator, Tuple
from gensim.models.phrases import Phraser
from spacy.language import Language
import gensim.corpora as corpora
import nltk
import spacy
from nltk.corpus import stopwords
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
import spacy
from sklearn.model_selection import train_test_split

load_dotenv()
data_dir = os.getenv('DATA_DIR')
tokenizer_dir = os.getenv('TOKENIZER_DIR')

#We should be really careful with the split, because date is really important

def load_data(url: str) -> pd.DataFrame:
    # Loading our data and sorting our dataframe based on date
    df = pd.read_csv(os.path.join(data_dir, url))

    return df

#create time periods function
#perform stratified sampling


def clean_data_sentiment(url: str, sample_size_per_month=10) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = load_data(url)

    df['published'] = pd.to_datetime(df['published'])

    # Sort by year first, then month, then day
    df_updated = df.sort_values(by='published', ascending=True)

    # Map sentiment values to new labels
    sentiment_mapping = {5: 1, 0: 2, -5: 0}  # Define the mapping: 5 -> Positive, 0 -> Neutral, -5 -> Negative
    df_updated['sentiment'] = df_updated['sentiment'].map(sentiment_mapping)


    df_updated['month_year'] = df_updated['published'].dt.to_period('M')

    # Use train_test_split with stratification on the 'monthly' column
    #train_set, test_set = train_test_split(df_updated, test_size=test_size, random_state=42)

    # Group by the 'month_year' column and sample records from each group
    grouped = df_updated.groupby('month_year')
    sampled_df = grouped.apply(lambda x: x.sample(min(len(x), sample_size_per_month), random_state=42)).reset_index(
        drop=True)

    #return train_set, test_set
    return sampled_df

def sent_to_words(sentences: List[str]) -> Generator[List[str], None, None]:
    """
    Convert a list of sentences into a generator that yields a list of words (tokens) for each sentence, removing punctuation.

    Args:
        sentences (List[str]): A list of sentences (strings).

    Yields:
        Generator[List[str], None, None]: A generator that yields a list of words (tokens) for each sentence.
    """
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def remove_stopwords(texts: List[List[str]], stop_words: List[str]) -> List[List[str]]:
    """
    Remove stopwords from the list of texts.

    Args:
        texts (List[List[str]]): A list of sentences (lists of strings).

    Returns:
        List[List[str]]: A list of sentences with stopwords removed.
    """
    return [[word for word in doc if word not in stop_words] for doc in texts]


def remove_stopwords_simple(sentences: List[str], stop_words: List[str]) -> List[str]:

    """
    Remove stopwords from a list of sentences.

    Args:
        sentences (List[str]): A list of sentences (strings).
        stop_words (List[str]): List of stopwords to be removed.

    Returns:
        List[str]: A list of sentences with stopwords removed.
    """

    return [' '.join([word for word in gensim.utils.simple_preprocess(str(sentence), deacc=True) if word not in stop_words]) for sentence in sentences]


def make_bigrams(texts: List[List[str]], bigram_mod: Phraser) -> List[List[str]]:
    """
    Convert sentences into bigrams.

    Args:
        texts (List[List[str]]): A list of tokenized sentences.
        bigram_mod (Phraser): A trained bigram model.

    Returns:
        List[List[str]]: A list of sentences with bigrams.
    """
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts: List[List[str]], trigram_mod: Phraser, bigram_mod: Phraser) -> List[List[str]]:
    """
    Convert sentences into trigrams.

    Args:
        texts (List[List[str]]): A list of tokenized sentences.
        trigram_mod (Phraser): A trained trigram model.
        bigram_mod (Phraser): A trained bigram model.

    Returns:
        List[List[str]]: A list of sentences with trigrams.
    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts: List[List[str]],
                  nlp: Language,
                  allowed_postags: List[str] = ['NOUN', 'ADJ', 'VERB', 'ADV']) -> List[List[str]]:
    """
    Perform lemmatization on the list of texts.

    Args:
        texts (List[List[str]]): A list of tokenized sentences.
        allowed_postags (List[str], optional): List of POS tags to keep. Defaults to ['NOUN', 'ADJ', 'VERB', 'ADV'].

    Returns:
        List[List[str]]: A list of lemmatized sentences.
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def filter_bigrams(bigrams: List[List[str]]) -> List[List[str]]:

    """
    Filters a list of bigrammed documents to include only those bigrams consisting of nouns and adjectives.

    Args:
        bigrams (List[List[str]]): A list of documents, where each document is a list of bigram strings.

    Returns:
        List[List[str]]: A filtered list of documents with bigrams containing only nouns and adjectives.
    """

    nlp = spacy.load("de_core_news_sm")

    filtered_bigrams = []
    for document in bigrams:
        filtered_doc = []
        for bigram in document:
            # Create a spaCy document for linguistic analysis
            doc = nlp(" ".join(bigram))
            # Filter out any bigram that contains verbs or adverbs
            if not any(token.pos_ in ['VERB', 'ADV'] for token in doc):
                filtered_doc.append(bigram)
        filtered_bigrams.append(filtered_doc)
    return filtered_bigrams

def remove_short_words(texts: List[List[str]]) -> List[List[str]]:
    """
    Removes words in each list of strings that are less than 4 characters long.

    Args:
        texts (List[List[str]]): A list of lists of words (bigrams).

    Returns:
        List[List[str]]: The same structure, with short words removed.
    """
    filtered_texts = []
    for text in texts:
        filtered_text = [' '.join(word for word in bigram.split() if len(word) > 4) for bigram in text]
        # Remove any empty strings that may result from removing short words
        filtered_texts.append([phrase for phrase in filtered_text if phrase])
    return filtered_texts


def preprocess_text_topic_modelling(df: pd.DataFrame,
                    col_needed: str,
                    sample_size=3000
                   ) -> Union[List[List[str]], List[Tuple[str, str]]]:
    """
    Preprocess text data based on the specified task.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        col_needed (str): Column name containing the raw text data.
        task_name (str): Name of the task. Choose from "topic_modelling" or "sentiment_analysis".
        label_col (str): Column name containing the labels (required for "sentiment_analysis").
        sample_size (int): Number of samples to take from the DataFrame. Default is 100.

    Returns:
        Union[List[List[str]], List[Tuple[str, str]]]: Processed text data based on the task.
            - For "topic_modelling": Returns a list of lists of bigrams.
            - For "sentiment_analysis": Returns a list of tuples with preprocessed documents and labels.
    """
    stop_words = stopwords.words('german')

    stop_words.extend(["aus", "ein", "und", "oder", "aber", "wenn", "dass", "mit", "für", "fur",
                       "null", "dpa", "bereits","gibt","ab", "steht", "gibt", "wurde", "sind",
                       "haben", "hat", "bei", "ab", "über", "von", "mit", "auf",
                       "für", "an", "im", "ist", "zu", "bis", "durch", "seit", "ohne", "gegen", "unter", "nach",
                       "aus", "am", "vor", "zur", "um", "als", "auch", "nicht", "noch", "werden", "mehr", "sehr",
                       "immer", "viele", "kann", "können", "ersten", "drei", "uhr", "online", "mussen", "ion", "sucht", "tag", "dabei",
                       "jahre", "tagen", "geworden","uhr","mussen","beim","ersten","drei","uber","konnten","adeg","geht","besonders","mal","nord","sud","to",
                       "open","up","on","ddrei","beide","sowie","la","dr","ml","of","hey","kg","jaap", "wahrend", "deutlich","konnen","rappen","braucht","konnte",
                       "deutschland","kommt","hallo","download","download it","harissa","gleich","prozent","it_pexels","butter"])

    # Step 1: Sample and clean the data

    df_cleaned = df[[col_needed]]
                  #.sample(n=sample_size, random_state=42))


    df_cleaned.dropna(inplace=True)
    df_cleaned['processed'] = df_cleaned[col_needed].map(lambda x: re.sub('\\n', ' ', x))
    df_cleaned['processed'] = df_cleaned['processed'].map(lambda x: x.lower())

    # Tokenize sentences
    sentences = df_cleaned['processed'].tolist()
    tokenized_sentences = list(sent_to_words(sentences))

    # Remove stopwords
    texts_no_stopwords = remove_stopwords(tokenized_sentences, stop_words)

    # Generate bigrams
    bigram = gensim.models.Phrases(texts_no_stopwords, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    texts_processed = make_bigrams(texts_no_stopwords, bigram_mod)

    #calling the function which will filter only names and adjectives
    filtered_bigrams = filter_bigrams(texts_processed)

    #calling the function which eleminates short words which are not significant for our analysis
    final_text_processed = remove_short_words(filtered_bigrams)

    return final_text_processed


def preprocess_text_sentiment_analysis(df: pd.DataFrame,
                    col_needed: str,
                    label_col: str = None,
                    date_col: str = None,
                    ) -> Union[List[List[str]], List[Tuple[str, str]]]:
    """
    Preprocess text data based on the specified task.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        col_needed (str): Column name containing the raw text data.
        task_name (str): Name of the task. Choose from "topic_modelling" or "sentiment_analysis".
        label_col (str): Column name containing the labels (required for "sentiment_analysis").
        sample_size (int): Number of samples to take from the DataFrame. Default is 100.

    Returns:
        Union[List[List[str]], List[Tuple[str, str]]]: Processed text data based on the task.
            - For "topic_modelling": Returns a list of lists of bigrams.
            - For "sentiment_analysis": Returns a list of tuples with preprocessed documents and labels.
    """
    stop_words = stopwords.words('german')


    df = df[[col_needed, label_col, date_col]]
    df.dropna(inplace=True)


    #can perform preprocessing steps such as lemmatization and so on
    df['processed'] = df[col_needed].map(lambda x: re.sub('\\n', ' ', x))
    df['processed'] = df['processed'].map(lambda x: x.lower())


    # Only basic preprocessing steps
    texts_no_stopwords = remove_stopwords_simple(df['processed'].tolist(), stop_words)

    # Generate bigrams
    #bigram = gensim.models.Phrases(texts_no_stopwords, min_count=5, threshold=100)
    #bigram_mod = Phraser(bigram)
    #texts_processed = make_bigrams(texts_no_stopwords, bigram_mod)

    # calling the function which will filter only names and adjectives
    #filtered_bigrams = filter_bigrams(texts_processed)

    # calling the function which eleminates short words which are not significant for our analysis
    #final_text_processed = remove_short_words(filtered_bigrams)

    labels = df[label_col].tolist()
    date = df[date_col].tolist()
    final_result = pd.DataFrame(zip(texts_no_stopwords, labels, date), columns=['Content','Label','Date'])

    return final_result

def convert_to_huggingface_datasets(data: List[Tuple[str, int]], train_ratio: float = 0.8, val_ratio: float = 0.2) -> DatasetDict:
    """
    Convert a list of tuples containing text and labels to a Hugging Face Dataset and split into train and validation sets.

    Args:
        data (List[Tuple[str, int]]): A list of tuples where each tuple contains a preprocessed sentence (str) and its corresponding label (int).
        train_ratio (float, optional): Ratio of data to allocate to the training set. Default is 0.8 (80% training).
        val_ratio (float, optional): Ratio of data to allocate to the validation set. Default is 0.2 (20% validation).

    Returns:
        DatasetDict: A Hugging Face DatasetDict object containing 'train' and 'val' splits.
    """

    # Calculate sizes for each split
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Create dictionaries for train and validation splits
    train_data_dict = {"text": [item[0] for item in data[:train_size]],
                       "label": [item[1] for item in data[:train_size]]}

    val_data_dict = {"text": [item[0] for item in data[train_size:train_size + val_size]],
                     "label": [item[1] for item in data[train_size:train_size + val_size]]}

    # Create Hugging Face DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_data_dict),
        "val": Dataset.from_dict(val_data_dict)
    })

    return dataset_dict

def convert_to_huggingface_datasets_v2(data: List[Tuple[str, int]],
                                       train_ratio: float = 0.6,
                                       val_ratio: float = 0.2,
                                       test_ratio: float = 0.2) -> DatasetDict:
    """
    Convert a single list of data tuples into Hugging Face datasets,
    and split into train, validation, and test datasets.

    Args:
        data (List[Tuple[str, int]]): A list of tuples, where each tuple contains a preprocessed sentence (str) and its corresponding label (int).
        train_ratio (float, optional): Ratio of the data to allocate to the training set.
        val_ratio (float, optional): Ratio of the data to allocate to the validation set.
        test_ratio (float, optional): Ratio of the data to allocate to the test set.

    Returns:
        DatasetDict: A Hugging Face DatasetDict object containing 'train', 'val', and 'test' splits.
    """
    # Ensure that the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be 1"

    # Shuffle data to ensure random distribution
    np.random.shuffle(data)

    # Calculate split indices
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Create dictionaries for each split
    train_data_dict = {"text": [item[0] for item in train_data],
                       "label": [item[1] for item in train_data]}
    val_data_dict = {"text": [item[0] for item in val_data],
                     "label": [item[1] for item in val_data]}
    test_data_dict = {"text": [item[0] for item in test_data],
                      "label": [item[1] for item in test_data],
                      "date":[item[2] for item in test_data]}

    # Create Hugging Face DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_data_dict),
        "val": Dataset.from_dict(val_data_dict),
        "test": Dataset.from_dict(test_data_dict)
    })

    return dataset_dict


def tokenize(example):

    # building the tokenizer and making the dataset ready for predicton
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer(example['text'], padding="max_length", truncation=True)

def return_tokenized_dataset(dataset, col_to_remove='text'):

    #mapping the changes to the dataset
    return (dataset.map(tokenize, batched=True, batch_size=None)
          .remove_columns(col_to_remove))

# Create Dictionary
def create_dictionary(data_words_bigrams: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:


    """
       Create a dictionary and corpus from lemmatized text data.

       Args:
           data_lemmatized (List[List[str]]): A list of lemmatized documents, where each document is a list of tokens.

       Returns:
           Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]: A tuple containing the Gensim dictionary object (`id2word`)
               and the corpus in the form of a list of bag-of-words representations of documents (`corpus`).
               - `id2word`: Gensim Dictionary object mapping words to their integer ids.
               - `corpus`: List of bag-of-words representations, where each element is a list of tuples (word_id, word_frequency)
                   representing the document.
       """

    id2word = corpora.Dictionary(data_words_bigrams)

    # Create Corpus
    texts = data_words_bigrams

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, corpus


if __name__ == "__main__":

    train_set, test_set = clean_data_sentiment('all_platforms_example.csv')

    train_data_sentiment = preprocess_text_sentiment_analysis(train_set, 'content', 'sentiment', 'test')

    print(train_data_sentiment.shape)