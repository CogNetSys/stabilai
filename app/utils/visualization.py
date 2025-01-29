# app/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot and return a confusion matrix figure.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    return fig


def plot_roc_curve(fpr, tpr, roc_auc, title='Receiver Operating Characteristic'):
    """
    Plot and return an ROC curve figure.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig


def plot_metrics(train_loss, val_loss, val_f1, title='Training Metrics'):
    """
    Plot training and validation loss and validation F1 score.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12,5))

    # Plot Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot F1 Score
    plt.subplot(1,2,2)
    plt.plot(epochs, val_f1, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Over Epochs')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('loss_f1_metrics.png')
    plt.close()