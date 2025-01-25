import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import calculate_entropy  # Ensure this is correctly implemented
from models.surge_collapse_net import SurgeCollapseNet  # Verify the architecture

def calculate_metrics(labels, preds, probs):
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {'f1': f1, 'precision': precision, 'recall': recall}

def plot_metrics(loss_history, entropy_history, val_loss_history, val_f1_history, title="Training Metrics"):
    epochs = range(1, len(loss_history) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot Entropy and F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, entropy_history, label='Train Entropy')
    plt.plot(epochs, val_f1_history, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Entropy / F1 Score')
    plt.title('Entropy and F1 Score Over Epochs')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_metrics.png')
    plt.show()

def train_model(
    model, train_loader, val_loader, optimizer, criterion, scheduler,
    num_epochs=100, device='cpu', early_stopping_patience=10, writer=None
):
    """
    Train the model.
    """
    model.to(device)
    logging.info(f"Training on device: {device}")
    best_f1 = 0.0
    epochs_no_improve = 0
    loss_history = []
    val_loss_history = []
    entropy_history = []
    val_f1_history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_entropy = 0.0

        # Training loop
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_entropy += calculate_entropy(targets)

        avg_loss = running_loss / len(train_loader)
        avg_entropy = total_entropy / len(train_loader)
        loss_history.append(avg_loss)
        entropy_history.append(avg_entropy)

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        val_f1_history.append(metrics['f1'])

        logging.info(
            f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Entropy={avg_entropy:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )

        # TensorBoard Logging
        if writer:
            writer.add_scalar('Loss/Train', avg_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Entropy/Train', avg_entropy, epoch)
            writer.add_scalar('F1_Score/Validation', metrics['f1'], epoch)
            writer.add_scalar('Precision/Validation', metrics['precision'], epoch)
            writer.add_scalar('Recall/Validation', metrics['recall'], epoch)

        # Learning Rate Scheduling
        scheduler.step(metrics['f1'])

        # Check for improvement
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            logging.info("Best model saved.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in F1-score for {epochs_no_improve} epoch(s).")

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logging.info("Training Complete.")
    plot_metrics(loss_history, entropy_history, val_loss_history, val_f1_history)
    if writer:
        writer.close()
    return loss_history, entropy_history, val_loss_history, val_f1_history

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )

    # Argument Parser
    parser = argparse.ArgumentParser(description="Train SurgeCollapseNet")
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--input_size', type=int, default=128, help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--output_size', type=int, default=2, help='Output size (number of classes)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    # Surge-Collapse parameters are omitted to disable them
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TensorBoard Writer
    writer = SummaryWriter(log_dir='runs/training_logs')

    # Synthetic Data
    def create_dummy_dataloader(num_samples, batch_size, input_size, output_size):
        # Create linearly separable data for binary classification
        inputs = torch.randn(num_samples, input_size)
        # Assign class 0 or 1 based on sum of inputs
        targets = (inputs.sum(dim=1) > 0).long()
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dummy_dataloader(4000, args.batch_size, args.input_size, args.output_size)
    val_loader = create_dummy_dataloader(1000, args.batch_size, args.input_size, args.output_size)

    # Model, Optimizer, and Criterion
    model = SurgeCollapseNet(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training
    train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=args.num_epochs, device=device,
        early_stopping_patience=args.early_stopping_patience,
        writer=writer
    )
