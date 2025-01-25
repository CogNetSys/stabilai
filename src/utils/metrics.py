from collections import Counter
import math
import torch
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
    Calculate evaluation metrics.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auroc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'auroc': auroc, 'ap': ap}
