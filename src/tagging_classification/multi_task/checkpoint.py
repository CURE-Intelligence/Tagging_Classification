import torch
import os
from src.tagging_classification.multi_task.multitask_model import XLMRobertaMultiTaskClassifier


def save_model(model, save_directory, optimizer=None, epoch=None, training_args=None):
    """Save the model and its components as a checkpoint."""

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Build the checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'config': {
            'num_article_types': model.article_type_classifier.out_features,
            'num_categories': model.category_classifier.out_features,
            'hidden_size': model.xlm_roberta.config.hidden_size,
            'model_type': 'xlm-roberta-multi-task'
        },
        'training_args': vars(training_args) if training_args else {}
    }

    # Save the checkpoint
    torch.save(checkpoint, os.path.join(save_directory, "multi_task_classifier_checkpoint.pth"))

    print(f"Model checkpoint and configuration saved in directory: {save_directory}")


# Loading the model (These methods are also located in checkpoint.py)
def load_model(save_directory, checkpoint_name, device='cpu'):
    """Load the model from a checkpoint for inference."""

    # Load the checkpoint
    checkpoint_path = os.path.join(save_directory, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration from the checkpoint
    config = checkpoint['config']

    # Recreate the model using the loaded configuration
    model = XLMRobertaMultiTaskClassifier(
        num_article_types=config['num_article_types'],
        num_categories=config['num_categories']
    ).to(device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    # Return the loaded model
    return model





