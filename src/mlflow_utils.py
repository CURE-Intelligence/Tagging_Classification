import mlflow
from visualizations import visualize_confusion_matrix


def log_confusion_matrix(y_true, y_pred):

    visualize_confusion_matrix(y_true, y_pred)
    mlflow.log_artifact("confusion_matrix.png")