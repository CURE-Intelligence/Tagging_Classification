from bertopic import BERTopic
from typing import List, Tuple
from bertopic import BERTopic
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def initialize_train_bert(sentences: List[str], n_topics: int = 10):

    # Load German stopwords from nltk
    #german_stop_words = stopwords.words('german')

    # Using a specific embedding model
    embedding_model = SentenceTransformer("T-Systems-onsite/german-roberta-sentence-transformer-v2")

    # Clustering model: HDBSCAN
    cluster_model = HDBSCAN(min_cluster_size=15,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)

    # Class TF-IDF Transformer with BM25 weighting
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)

    # Custom vectorizer: Stemmed Count Vectorizer
    vectorizer_model = CountVectorizer(analyzer="word",
                                              ngram_range=(1, 2))

    # BERTopic model setup with the above models
    topic_model = BERTopic(embedding_model=embedding_model,
                           hdbscan_model=cluster_model,
                           ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model,
                           language="german")

    topics, probs = topic_model.fit_transform(sentences)

    # Save the trained model to the specified path
    #we can do this later (when wanting to predict in inference)
    #topic_model.save(model_path)
    return topic_model, topics, probs

#This can be done at a second moment (we don't need to have stored the model somewhere)
def load_bert(model_path):

    # Load a BERTopic model from a saved file
    topic_model = BERTopic.load(model_path)
    return topic_model





