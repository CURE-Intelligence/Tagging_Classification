import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from tqdm.auto import tqdm
import pandas as pd
import os
import random

SUPPORTED_LANGUAGES = ['de', 'en', 'es', 'zh-cn', 'it', 'fr', 'nl', 'sv', 'vi', 'zh-tw']

# Dictionary mapping languages to appropriate models
LANGUAGE_MODELS = {
    'de': 'bert-base-german-cased',
    'en': 'bert-base-uncased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'zh-cn': 'bert-base-chinese',
    'it': 'dbmdz/bert-base-italian-cased',
    'fr': 'camembert-base',
    'nl': 'wietsedv/bert-base-dutch-cased',
    'sv': 'KB/bert-base-swedish-cased',
    'vi': 'vinai/phobert-base',
    'zh-tw': 'bert-base-chinese'  # Using the same model for Traditional Chinese
}


def get_augmenters(lang):
    """
    Get a list of text augmenters for a given language.

    Args:
        lang (str): The language code.

    Returns:
        list: A list of nlpaug augmenters suitable for the given language.
    """
    model_name = LANGUAGE_MODELS.get(lang, 'xlm-roberta-base')
    return [
        naw.ContextualWordEmbsAug(model_path=model_name, action="substitute"),
        naw.RandomWordAug(action="swap"),
        nac.RandomCharAug(action="substitute")
    ]


def augment_text(text, lang, num_augmented=1):
    """
    Augment a single text using language-specific augmenters.

    Args:
        text (str): The text to augment.
        lang (str): The language of the text.
        num_augmented (int): Number of augmented versions to generate.

    Returns:
        list: A list of augmented texts.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    augmenters = get_augmenters(lang)
    augmented_texts = []
    for _ in range(num_augmented):
        aug = random.choice(augmenters)
        try:
            augmented = aug.augment(text)
            if isinstance(augmented, list):
                augmented = augmented[0]
            augmented_texts.append(augmented)
        except Exception as e:
            print(f"Augmentation error for text: {text[:100]}... Error: {e}")
    return augmented_texts


def calculate_samples_to_add(df, article_type_column, categorie_column, target_size):

    """
    Calculate the number of samples to add for each combination of article_type and categorie.

    Args:
        df (pd.DataFrame): Original DataFrame.
        article_type_column (str): Name of the column containing article types.
        categorie_column (str): Name of the column containing categories.
        target_size (int): The desired size of the final dataset.

    Returns:
        dict: A dictionary with (article_type, categorie) tuples as keys and number of samples to add as values.
    """


    current_counts = df.groupby([article_type_column, categorie_column]).size()
    total_combinations = len(current_counts)
    target_count_per_combination = target_size // total_combinations

    samples_to_add = {}
    for (article_type, categorie), count in current_counts.items():
        samples_to_add[(article_type, categorie)] = max(0, target_count_per_combination - count)

    return samples_to_add


def augment_combination(texts, langs, article_type, categorie, samples_to_add):
    """
    Augment texts for a specific combination of article_type and categorie.

    Args:
        texts (list): Original texts of the combination.
        langs (list): Languages of the texts.
        article_type (str): The article type.
        categorie (str): The category.
        samples_to_add (int): Number of samples to add for this combination.

    Returns:
        tuple: Lists of augmented texts, their article types, and categories.
    """
    augmented_texts = []
    augmented_article_types = []
    augmented_categories = []

    with tqdm(total=samples_to_add, desc=f"Augmenting {article_type} - {categorie}") as pbar:
        while len(augmented_texts) < samples_to_add:
            for text, lang in zip(texts, langs):
                if len(augmented_texts) >= samples_to_add:
                    break
                augmented = augment_text(text, lang)
                if augmented:
                    augmented_texts.extend(augmented)
                    augmented_article_types.extend([article_type])
                    augmented_categories.extend([categorie])
                    pbar.update(1)

    return augmented_texts, augmented_article_types, augmented_categories



def create_augmented_dataframe(df, text_column, article_type_column, categorie_column, lang_column, samples_to_add):
    """
    Create a DataFrame of augmented data for all combinations of article_type and categorie.

    Args:
        df (pd.DataFrame): Original DataFrame.
        text_column (str): Name of the column containing text data.
        article_type_column (str): Name of the column containing article types.
        categorie_column (str): Name of the column containing categories.
        lang_column (str): Name of the column containing language information.
        samples_to_add (dict): Dictionary specifying how many samples to add for each combination.

    Returns:
        pd.DataFrame: A DataFrame containing the augmented data.
    """
    all_augmented_texts = []
    all_augmented_article_types = []
    all_augmented_categories = []

    for (article_type, categorie), to_add in samples_to_add.items():
        subset = df[(df[article_type_column] == article_type) & (df[categorie_column] == categorie)]
        texts = subset[text_column].tolist()
        langs = subset[lang_column].tolist()

        aug_texts, aug_types, aug_categories = augment_combination(texts, langs, article_type, categorie, to_add)

        all_augmented_texts.extend(aug_texts)
        all_augmented_article_types.extend(aug_types)
        all_augmented_categories.extend(aug_categories)

    return pd.DataFrame({
        text_column: all_augmented_texts,
        article_type_column: all_augmented_article_types,
        categorie_column: all_augmented_categories,
        lang_column: [df[lang_column].iloc[0]] * len(all_augmented_texts)
    })


def balanced_augmentation(df, text_column, article_type_column, categorie_column, lang_column,
                          target_size=32000, output_folder='balanced_augmented_data'):
    """
    Perform balanced augmentation on the dataset to achieve a target size with balanced classes
    for both article_type and categorie.

    Args:
        df (pd.DataFrame): Original DataFrame.
        text_column (str): Name of the column containing text data.
        article_type_column (str): Name of the column containing article types.
        categorie_column (str): Name of the column containing categories.
        lang_column (str): Name of the column containing language information.
        target_size (int): The desired size of the final dataset.
        output_folder (str): Folder to save the augmented dataset.

    Returns:
        pd.DataFrame: The balanced and augmented DataFrame.
    """
    original_size = len(df)
    samples_to_add = calculate_samples_to_add(df, article_type_column, categorie_column, target_size)

    augmented_df = create_augmented_dataframe(df, text_column, article_type_column, categorie_column, lang_column,
                                              samples_to_add)

    result_df = pd.concat([df, augmented_df], ignore_index=True)

    if len(result_df) > target_size:
        result_df = result_df.sample(n=target_size, random_state=42)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'augmented_dataset_multi_v2.csv')
    result_df.to_csv(output_file, index=False)

    print(f"Balanced augmented dataset saved to {output_file}")
    print(f"Original size: {original_size}")
    print(f"New size: {len(result_df)}")
    print("\nArticle Type distribution before augmentation:")
    print(df[article_type_column].value_counts())
    print("\nArticle Type distribution after augmentation:")
    print(result_df[article_type_column].value_counts())
    print("\nCategorie distribution before augmentation:")
    print(df[categorie_column].value_counts())
    print("\nCategorie distribution after augmentation:")
    print(result_df[categorie_column].value_counts())
    print("\nCombined (Article Type, Categorie) distribution after augmentation:")
    print(result_df.groupby([article_type_column, categorie_column]).size())

    return result_df