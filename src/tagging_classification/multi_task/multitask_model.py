import torch
from torch import nn
from transformers import XLMRobertaModel, PreTrainedModel, XLMRobertaConfig
import os
import json

class XLMRobertaMultiTaskClassifier(nn.Module):

#class XLMRobertaMultiTaskClassifier(PreTrainedModel):

    #config_class = XLMRobertaConfig


    """
    -Class Definition
    -Class Name: XLMRobertaMultiTaskClassifier
        Base Class: nn.Module (PyTorch's base class for all neural network modules)
        Initialization (__init__ method)
        The constructor (__init__) is meant to initialize the attributes of the class. However, there is a typo in the method name; it should be __init__ with double underscores before and after init.

    Parameters:

        num_article_types: Number of classes for the article type classification task.
        num_categories: Number of classes for the category classification task.
        Super Constructor: super(XLMRobertaMultiTaskClassifier, self).__init__() properly initializes the base class. However, it’s incorrectly typed as _init_ in the code.

    XLM-RoBERTa Model:

        self.xlm_roberta: An instance of the XLMRobertaModel pre-trained model from Hugging Face's transformers library. It’s loaded with the base version of the model.
        Dropout Layer:

        self.dropout: A dropout layer with a dropout rate of 0.1 to prevent overfitting.
    Classifiers:

        self.article_type_classifier: A linear layer that maps the hidden size of the XLM-RoBERTa model to the number of article types.
        self.category_classifier: A linear layer that maps the hidden size to the number of categories.
    """

    def __init__(self, num_article_types, num_categories):
        super(XLMRobertaMultiTaskClassifier, self).__init__()

        # It loads the 'xlm-roberta-base' model directly, ensuring you get the exact pre-trained weights without any modifications.
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(p=0.1)
        self.article_type_classifier = nn.Linear(self.xlm_roberta.config.hidden_size, num_article_types)
        self.category_classifier = nn.Linear(self.xlm_roberta.config.hidden_size, num_categories)



    def forward(self, **inputs):
        """
        The forward method defines the forward pass of the model.

        Inputs:

            input_ids: Input token IDs.
            attention_mask: Mask to avoid performing attention on padding token indices.
            labels: (Optional) A dictionary containing labels for both classification tasks.
            XLM-RoBERTa Output:

            outputs: The output of the XLM-RoBERTa model.
            pooled_output: The representation of the [CLS] token, which captures the aggregated information of the input sequence.
        Dropout:

            The pooled output is passed through the dropout layer.
        Classifiers:

            article_type_logits: Logits for the article type classification.
            category_logits: Logits for the category classification.
        Loss Calculation:

            If labels are provided, the loss is calculated using nn.CrossEntropyLoss() for both tasks and summed.
        Return:

            A dictionary containing the total loss (if labels are provided), and the logits for both classification tasks.


        """

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.get('labels', None)  # Use get to handle cases where labels might not be provided

        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        #The line pooled_output = outputs.last_hidden_state[:, 0, :] extracts
        #the hidden state of the[CLS] token for each sequence in the batch.
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        article_type_logits = self.article_type_classifier(pooled_output)
        category_logits = self.category_classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            article_type_loss = loss_fct(article_type_logits, labels['article_type'])
            category_loss = loss_fct(category_logits, labels['categorie'])
            loss = article_type_loss + category_loss

        return {'loss': loss, 'article_type_logits': article_type_logits, 'category_logits': category_logits}



