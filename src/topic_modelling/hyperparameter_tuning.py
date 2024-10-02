#Performing Hyperparameter Tuning

"""
First, let's differentiate between model hyperparameters and model parameters :
Model hyperparameters can be thought of as settings for a machine learning algorithm that are tuned by the data scientist before training. Examples would be the number of trees in the random forest, or in our case, number of topics K
Model parameters can be thought of as what the model learns during training, such as the weights for each word in a given topic.
Now that we have the baseline coherence score for the default LDA model, let's perform a series of sensitivity tests to help determine the following model hyperparameters:
Number of Topics (K) Dirichlet hyperparameter alpha: Document-Topic Density Dirichlet hyperparameter beta: Word-Topic Density

"""

import numpy as np
import tqdm
import pandas as pd
import gensim
import gensim.corpora as corpora
from typing import List, Tuple, Any, Dict, Union
from src.metrics import compute_coherence
from src.topic_modelling.lda_base_model import initialize_model

def hyperparameter_tuning_lda(corpus: List[List[Tuple[int, int]]],
                          id2word: corpora.Dictionary,
                          data: List[List[str]],
                          tuning_params: Dict[str, List[Union[int, float, str]]]) -> pd.DataFrame:
    """
    Perform hyperparameter tuning for LDA model using coherence score evaluation.

    Args:
        corpus (List[List[Tuple[int, int]]]): A list of bag-of-words representations of documents.
            Each document is represented as a list of tuples (word_id, word_frequency).
        id2word (corpora.Dictionary): Gensim dictionary mapping words to their integer ids.
        data_lemmatized (List[List[str]]): Lemmatized documents.
        tuning_params (Dict[str, List[Union[int, float, str]]]): Dictionary containing lists of tuning parameters including
            'alpha_values', 'beta_values', 'topics_range', and 'corpus_titles'.

    Returns:
        pd.DataFrame: DataFrame containing tuning results with columns 'Validation_Set', 'Topics', 'Alpha', 'Beta', 'Coherence'.
    """
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []}

    pbar = tqdm.tqdm(total=(len(tuning_params['beta_values']) * len(tuning_params['alpha_values']) *
                            len(tuning_params['topics_range']) * len(tuning_params['corpus_titles'])))

    # Iterate through validation corpuses
    for i, corpus_set in enumerate([gensim.utils.ClippedCorpus(corpus, int(len(corpus) * 0.75)), corpus]):
        for corpus_title in tuning_params['corpus_titles']:
            for num_topics in tuning_params['topics_range']:
                for alpha in tuning_params['alpha_values']:
                    for beta in tuning_params['beta_values']:
                        model_params = {'num_topics': num_topics, 'alpha': alpha, 'eta': beta}
                        lda_model = initialize_model(corpus_set, id2word, model_params)
                        coherence = compute_coherence(lda_model, data, id2word)

                        model_results['Validation_Set'].append(corpus_title)
                        model_results['Topics'].append(num_topics)
                        model_results['Alpha'].append(alpha)
                        model_results['Beta'].append(beta)
                        model_results['Coherence'].append(coherence)

                        pbar.update(1)

    pbar.close()
    return pd.DataFrame(model_results)


"""
def hyperparameter_tuning_bertopic(data: List[str], parameter_grid: Dict[str, List[Any]]) -> pd.DataFrame:

    
       Perform hyperparameter tuning for BERTopic.

       Args:
           data (List[str]): List of preprocessed documents.
           parameter_grid (Dict[str, List[Any]]): Dictionary containing lists of tuning parameters.

       Returns:
           pd.DataFrame: DataFrame containing tuning results with columns 'n_topics', 'nr_topics', 'min_cluster_size',
                         'nr_representatives', 'topic_similarity_threshold', 'coherence'.
       
    tuning_results = []

    for n_topics in parameter_grid['n_topics']:
        for nr_topics in parameter_grid['nr_topics']:
            for min_cluster_size in parameter_grid['min_cluster_size']:
                for nr_representatives in parameter_grid['nr_representatives']:
                    for topic_similarity_threshold in parameter_grid['topic_similarity_threshold']:
                        # Initialize BERTopic with current parameters
                        topic_model = initialize_bertopic_model(n_topics=n_topics,
                                               nr_topics=nr_topics,
                                               min_cluster_size=min_cluster_size,
                                               nr_representatives=nr_representatives,
                                               topic_similarity_threshold=topic_similarity_threshold)

                        # Fit BERTopic on data
                        topics, _ = topic_model.fit_transform(data)

                        # Calculate coherence score (if applicable)
                        #we can also add compute_coherence from the function defined in metrics.py
                        #coherence = topic_model.get_coherence()
                        coherence = compute_coherence(topic_model, data, dictionary=None, model_name="BERTopic")

                        # Store results
                        tuning_results.append({
                            'n_topics': n_topics,
                            'nr_topics': nr_topics,
                            'min_cluster_size': min_cluster_size,
                            'nr_representatives': nr_representatives,
                            'topic_similarity_threshold': topic_similarity_threshold,
                            'coherence': coherence
                        })

    return pd.DataFrame(tuning_results)
"""