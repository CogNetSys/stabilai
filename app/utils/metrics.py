# app/utils/metrics.py

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc

# metrics.py

def calculate_metrics(labels, preds, probs):
    """
    Compute classification metrics for binary classification.
    """
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {'f1': f1, 'precision': precision, 'recall': recall}

def run_eval(model, loader, criterion, device):
    """
    Evaluate the model on the given DataLoader.
    Returns average loss, f1, precision, recall, confusion matrix, classification report, FPR, TPR, ROC AUC.
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    f1, precision, recall = calculate_metrics(all_labels, all_preds, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        # Handle cases with only one class in y_true
        fpr, tpr, roc_auc = [0.0, 1.0], [0.0, 1.0], 0.0
    
    return avg_loss, f1, precision, recall, cm, report, fpr, tpr, roc_auc
