import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding



class MultiTaskDataCollator:
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([torch.tensor(f['input_ids']) for f in features])
        batch['attention_mask'] = torch.stack([torch.tensor(f['attention_mask']) for f in features])
        batch['labels'] = {
            'article_type': torch.tensor([f['labels']['article_type'] for f in features]),
            'categorie': torch.tensor([f['labels']['categorie'] for f in features])
        }
        return batch


"""
class MultiTaskDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        batch = self.data_collator(features)

        # Handle labels
        if "article_type" in features[0] and "category" in features[0]:
            batch["labels"] = {
                "article_type": torch.tensor([f["article_type"] for f in features], dtype=torch.long),
                "category": torch.tensor([f["category"] for f in features], dtype=torch.long)
            }
        else:
            print("Warning: 'article_type' or 'category' not found in features. This might cause issues during training.")

        return batch
        
"""