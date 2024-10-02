from typing import List, Tuple, Union
import gensim
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
import gensim.corpora as corpora
import numpy as np

from typing import List
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from bertopic import BERTopic


def compute_coherence(model,
                      texts: List[str],
                      dictionary: Dictionary,
                      model_name: str) -> float:
    """
    Compute coherence score for the specified model.

    Args:
        model: The initialized model (either LDA or BERTopic).
        texts (List): Preprocessed texts (list of documents).
        dictionary (Dictionary): Gensim dictionary mapping words to their integer ids.
        model_name (str): Name of the model ('LDA' or 'BERTopic').

    Returns:
        float: Coherence score.
    """

    if model_name == "LDA":
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        return coherence_model.get_coherence()
    elif model_name == "BERTopic":
        coherence_model_bertopic = model.coherence_model(texts=texts, coherence='c_v')
        return coherence_model_bertopic.get_coherence()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")



def compute_silhouette(model: Union[BERTopic, gensim.models.LdaMulticore],
                       documents_or_corpus: Union[List[str], List[List[Tuple[int, int]]]],
                       embeddings: List[np.ndarray] = None,
                       model_name: str = "LDA",
                       n_neighbors: int = 15) -> float:

    """
    Compute silhouette score for document clustering based on the given topic model (BERTopic or LDA).

    Args:
        model (Union[BERTopic, gensim.models.LdaMulticore]): Initialized topic model.
        documents_or_corpus (Union[List[str], List[List[Tuple[int, int]]]]): List of documents (strings) for BERTopic or corpus in bag-of-words format for LDA.
        embeddings (List[np.ndarray], optional): List of document embeddings (e.g., BERT embeddings) for BERTopic. Default is None.
        model_name (str): Name of the model ("BERTopic" or "LDA").
        n_neighbors (int, optional): Number of neighbors to consider for silhouette score calculation. Default is 15.

    Returns:
        float: Silhouette score.
    """

    if model_name == "BERTopic":
        if embeddings is None:
            raise ValueError("Embeddings are required for BERTopic.")
        # Fit BERTopic model and transform documents to clusters
        _, _ = model.fit_transform(documents_or_corpus, embeddings)
        labels = model.get_labels()
        return silhouette_score(embeddings, labels, metric='cosine')
    elif model_name == "LDA":
        # Assign documents to topics
        topic_assignments = [max(model[doc], key=lambda x: x[1])[0] for doc in documents_or_corpus]
        return silhouette_score(documents_or_corpus, topic_assignments)
    else:
        raise ValueError("Invalid model name. Choose 'BERTopic' or 'LDA'.")

# Example usage:
# For BERTopic
# bertopic_model = BERTopic(...)
# bertopic_documents = ["document1", "document2", ...]  # List of documents (strings)
# bertopic_embeddings = [embedding1, embedding2, ...]  # List of document embeddings
# silhouette_score_bertopic = compute_silhouette(bertopic_model, bertopic_documents, embeddings=bertopic_embeddings, model_name="BERTopic")

# For LDA
# lda_model = gensim.models.LdaMulticore(...)
# lda_corpus = [[(0, 1), (1, 2)], [(2, 1), (3, 1)], ...]  # Corpus in bag-of-words format
# silhouette_score_lda = compute_silhouette(lda_model, lda_corpus, model_name="LDA")

