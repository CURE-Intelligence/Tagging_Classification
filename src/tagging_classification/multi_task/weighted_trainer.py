import torch


from transformers import Trainer

"""
class WeightedLossTrainer(Trainer):
  def __init__(self, *args, class_weights, **kwargs):
    super().__init__(*args, **kwargs)
    # Don't move to device here, we'll do it in compute_loss
    self.class_weights = class_weights

  def compute_loss(self, model, inputs, return_outputs=False):
    # Move inputs and class weights to the model's device

    # Forward pass
    outputs = model(**inputs)
    preds = outputs.logits
    labels = inputs.get("labels")

    # Ensure preds and labels have the correct shape
    loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights)
    loss = loss_func(preds, labels)

    return (loss, outputs) if return_outputs else loss
    
"""

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, article_type_class_weights, category_class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.article_type_class_weights = article_type_class_weights
        self.category_class_weights = category_class_weights

    def compute_loss(self, model, inputs, return_outputs=False):

        # Forward pass
        outputs = model(**inputs)
        preds_article_type = outputs['article_type_logits']
        preds_category = outputs['category_logits']
        labels = inputs.get("labels")

        # Extract labels for each task
        labels_article_type = labels['article_type']
        labels_category = labels['categorie']

        # Compute losses
        loss_func_article_type = torch.nn.CrossEntropyLoss(weight=self.article_type_class_weights)
        loss_func_category = torch.nn.CrossEntropyLoss(weight=self.category_class_weights)

        # Compute losses using FocalLoss with class weights
        #loss_func_article_type = FocalLoss(gamma=2, weight=self.article_type_class_weights)
        #loss_func_category = FocalLoss(gamma=2, weight=self.category_class_weights)

        loss_article_type = loss_func_article_type(preds_article_type, labels_article_type)
        loss_category = loss_func_category(preds_category, labels_category)

        # Combine losses (simple sum in this case)
        loss = loss_article_type + loss_category

        return (loss, outputs) if return_outputs else loss