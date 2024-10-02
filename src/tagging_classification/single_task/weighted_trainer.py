import torch
from transformers import Trainer

class WeightedLossTrainerSingle(Trainer):

    """Will create the WeightedLoss for the single task classification problem"""
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        preds = outputs['logits']  # Assuming 'logits' contains the model's predictions
        labels = inputs.get("labels")

        # Compute the loss
        loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(preds, labels)

        return (loss, outputs) if return_outputs else loss