import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

def plot_confusion_matrix(labels, preds, title='Confusion Matrix'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_precision_recall(labels, probs, title='Precision-Recall Curve'):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_metrics(loss_history, entropy_history, title="Training Metrics"):
    epochs = range(1, len(loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(12,6))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_history, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Entropy', color=color)
    ax2.plot(epochs, entropy_history, color=color, label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.show()
