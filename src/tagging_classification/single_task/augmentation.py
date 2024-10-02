from src.tagging_classification.multi_task.augmentation import SUPPORTED_LANGUAGES, LANGUAGE_MODELS,get_augmenters, augment_text
import os
from tqdm import tqdm  # Change this line
import pandas as pd

def calculate_samples_to_add(df, target_column, target_size):
    """
    Calculate the number of samples to add for each class in the target column.

    Args:
        df (pd.DataFrame): Original DataFrame.
        target_column (str): Name of the column containing the target variable.
        target_size (int): The desired size of the final dataset.

    Returns:
        dict: A dictionary with target classes as keys and number of samples to add as values.
    """
    current_counts = df[target_column].value_counts()
    total_classes = len(current_counts)
    target_count_per_class = target_size // total_classes

    samples_to_add = {}
    for target_class, count in current_counts.items():
        samples_to_add[target_class] = max(0, target_count_per_class - count)

    return samples_to_add

def augment_class(texts, langs, target_class, samples_to_add):
    """
    Augment texts for a specific target class.

    Args:
        texts (list): Original texts of the class.
        langs (list): Languages of the texts.
        target_class (str): The target class.
        samples_to_add (int): Number of samples to add for this class.

    Returns:
        tuple: Lists of augmented texts and their target classes.
    """
    augmented_texts = []
    augmented_targets = []

    with tqdm(total=samples_to_add, desc=f"Augmenting class {target_class}") as pbar:
        while len(augmented_texts) < samples_to_add:
            for text, lang in zip(texts, langs):
                if len(augmented_texts) >= samples_to_add:
                    break
                augmented = augment_text(text, lang)
                if augmented:
                    augmented_texts.extend(augmented)
                    augmented_targets.extend([target_class])
                    pbar.update(1)

    return augmented_texts, augmented_targets

def create_augmented_dataframe(df, text_column, target_column, lang_column, samples_to_add):
    """
    Create a DataFrame of augmented data for all classes in the target column.

    Args:
        df (pd.DataFrame): Original DataFrame.
        text_column (str): Name of the column containing text data.
        target_column (str): Name of the column containing the target variable.
        lang_column (str): Name of the column containing language information.
        samples_to_add (dict): Dictionary specifying how many samples to add for each class.

    Returns:
        pd.DataFrame: A DataFrame containing the augmented data.
    """
    all_augmented_texts = []
    all_augmented_targets = []

    for target_class, to_add in samples_to_add.items():
        subset = df[df[target_column] == target_class]
        texts = subset[text_column].tolist()
        langs = subset[lang_column].tolist()

        aug_texts, aug_targets = augment_class(texts, langs, target_class, to_add)

        all_augmented_texts.extend(aug_texts)
        all_augmented_targets.extend(aug_targets)

    return pd.DataFrame({
        text_column: all_augmented_texts,
        target_column: all_augmented_targets,
        lang_column: [df[lang_column].iloc[0]] * len(all_augmented_texts)
    })

def balanced_augmentation(df, text_column, target_column, lang_column,
                          target_size=32000, output_folder='balanced_augmented_data'):
    """
    Perform balanced augmentation on the dataset to achieve a target size with balanced classes
    for the target variable.

    Args:
        df (pd.DataFrame): Original DataFrame.
        text_column (str): Name of the column containing text data.
        target_column (str): Name of the column containing the target variable.
        lang_column (str): Name of the column containing language information.
        target_size (int): The desired size of the final dataset.
        output_folder (str): Folder to save the augmented dataset.

    Returns:
        pd.DataFrame: The balanced and augmented DataFrame.
    """
    original_size = len(df)
    samples_to_add = calculate_samples_to_add(df, target_column, target_size)

    augmented_df = create_augmented_dataframe(df, text_column, target_column, lang_column, samples_to_add)

    result_df = pd.concat([df, augmented_df], ignore_index=True)

    if len(result_df) > target_size:
        result_df = result_df.sample(n=target_size, random_state=42)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'augmented_dataset_single_v2.csv')
    result_df.to_csv(output_file, index=False)

    print(f"Balanced augmented dataset saved to {output_file}")
    print(f"Original size: {original_size}")
    print(f"New size: {len(result_df)}")
    print("\nTarget variable distribution before augmentation:")
    print(df[target_column].value_counts())
    print("\nTarget variable distribution after augmentation:")
    print(result_df[target_column].value_counts())

    return result_df