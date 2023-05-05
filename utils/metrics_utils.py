import numpy as np
import sklearn.metrics as metrics

def get_confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate the confusion matrix.

    Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.
    num_classes: Number of classes in the dataset.

    Returns:
    A numpy array representing the confusion matrix.
    """
    return metrics.confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

def get_classification_report(y_true, y_pred, num_classes, target_names=None):
    """
    Calculate the classification report including precision, recall, and F1-score.

    Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.
    num_classes: Number of classes in the dataset.
    target_names: Optional list of class names.

    Returns:
    A string representation of the classification report.
    """
    return metrics.classification_report(y_true, y_pred, labels=np.arange(num_classes), target_names=target_names)

def get_accuracy(y_true, y_pred):
    """
    Calculate the accuracy.

    Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.

    Returns:
    float: The accuracy.
    """
    return metrics.accuracy_score(y_true, y_pred)

def get_precision_recall_f1(y_true, y_pred, average='macro'):
    """
    Calculate precision, recall, and F1-score.

    Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.
    average: The method to calculate the average score ('micro', 'macro', or 'weighted').

    Returns:
    tuple: (precision, recall, f1_score)
    """
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average=average)
    f1_score = metrics.f1_score(y_true, y_pred, average=average)
    
    return precision, recall, f1_score
