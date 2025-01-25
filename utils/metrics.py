# utils/metrics.py

import math
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def calculate_entropy(targets):
    """
    Calculate entropy of target labels.
    """
    counts = Counter(targets.tolist())
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p + 1e-10) for p in probabilities)
    return entropy

def calculate_metrics(labels, preds, probs):
    """
    Calculate evaluation metrics for multiclass or binary classification.
    Handles both cases dynamically.
    """
    if len(set(labels)) > 2:  # Multiclass case
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': None,  # AUROC is not well-defined for multiclass
            'ap': None  # Average precision is also not well-defined for multiclass
        }
    else:  # Binary classification case
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        auroc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'ap': ap
        }

    return metrics