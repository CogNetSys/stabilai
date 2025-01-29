# app/training.py

import torch
import torch.nn as nn
from app.utils.optimizer import OrthogonalGradientOptimizer
from app.utils.losses import StableMaxCrossEntropy
from app.utils.metrics import run_eval
from app.utils.visualization import plot_confusion_matrix, plot_roc_curve
from app.utils.gradfilter import gradfilter_ma, gradfilter_ema
from app.data.data_loader import add_gaussian_noise
import logging
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def reset_and_inject_noise(model, grad_norms, threshold, noise_level):
    """
    Identify parameters with gradient norms below the threshold,
    reset their weights, and inject Gaussian noise.

    Args:
        model (nn.Module): The neural network model.
        grad_norms (dict): Dictionary of gradient norms per parameter.
        threshold (float): Threshold below which gradients are considered stale.
        noise_level (float): Standard deviation of the Gaussian noise to inject.

    Returns:
        reset_params (list): List of parameter names that were reset.
    """
    reset_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and name in grad_norms:
            if grad_norms[name] < threshold:
                # Reset the weights
                nn.init.zeros_(param)
                # Inject Gaussian noise
                noise = torch.randn_like(param) * noise_level
                param.data += noise
                reset_params.append(name)
    return reset_params


def calculate_activation_entropy(activations):
    """
    Calculate entropy for each layer's activations.

    Args:
        activations (dict): Dictionary of activations per layer.

    Returns:
        entropy_dict (dict): Dictionary of entropy values per layer.
    """
    entropy_dict = {}
    for layer_name, activation in activations.items():
        # Apply softmax to activations to get probability distributions
        probs = nn.functional.softmax(activation, dim=1)
        # Calculate entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
        entropy_dict[layer_name] = entropy
    return entropy_dict


def calculate_gradient_norms(model):
    """
    Calculate the L2 norm of gradients for each parameter in the model.

    Returns:
        grad_norms (dict): Dictionary mapping parameter names to their gradient norms.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[name] = grad_norm
    return grad_norms


def train_fastgrokkingrush(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    early_stopping_patience=10,
    writer=None,
    use_grokfast=False,
    grokfast_type='ema',
    alpha=0.98,
    lamb=2.0,
    window_size=100,
    filter_type='mean',
    warmup=True,
    gradient_threshold=1e-3,
    noise_level=1e-3,
    dataset_noise_level=0.0,
    entropy_threshold=1.5,
    run_dir="."
):
    """
    Comprehensive training loop with Surge-Collapse, Entropy-Based Diagnostics,
    Learning Rate Scheduler, Enhanced Early Stopping, and Model Checkpointing.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to train on ('cuda' or 'cpu').
        epochs (int): Number of training epochs.
        early_stopping_patience (int): Early stopping patience.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging.
        use_grokfast (bool): Whether to use Grokfast gradient filtering.
        grokfast_type (str): Type of Grokfast filtering ('ema' or 'ma').
        alpha (float): EMA momentum.
        lamb (float): Amplification factor.
        window_size (int): Window size for MA.
        filter_type (str): Filter type for MA ('mean' or 'sum').
        warmup (bool): Whether to enable warmup for MA.
        gradient_threshold (float): Threshold for stale gradients.
        noise_level (float): Noise level for injected noise.
        dataset_noise_level (float): Dataset noise injection level.
        entropy_threshold (float): Entropy threshold for Surge-Collapse.
        run_dir (str): Directory to save logs and models.

    Returns:
        history (dict): Dictionary containing training history.
    """
    model.to(device)
    logging.info(f"[FastGrokkingRush] Training on device={device}...")

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    lr_adjustments = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'layer_activation_entropy': {},
        'reset_params': [],
        'best_epoch': 0,
        'lr_adjustments': 0,
        'classification_report': ""
    }

    # Initialize Grokfast gradient storage
    if use_grokfast and grokfast_type == 'ema':
        grads = None  # Will be initialized in gradfilter_ema
    elif use_grokfast and grokfast_type == 'ma':
        grads = None  # Will be initialized in gradfilter_ma
    else:
        grads = None  # Not used

    # Initialize Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Initialize Early Stopping
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        # Dictionaries to accumulate layer-wise stats
        epoch_activation_entropy = {}

        # Training pass
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            # Inject noise into dataset if applicable
            if dataset_noise_level > 0.0:
                Xb = add_gaussian_noise(Xb, noise_level=dataset_noise_level)

            optimizer.zero_grad()

            logits, activations = model(Xb)  # Capture activations
            loss = criterion(logits, yb)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Apply Grokfast gradient filtering if enabled
            if use_grokfast:
                if grokfast_type == 'ema':
                    grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)
                elif grokfast_type == 'ma':
                    grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb, filter_type=filter_type, warmup=warmup)
                else:
                    raise ValueError(f"Unsupported grokfast_type: {grokfast_type}")

            optimizer.step()

            running_loss += loss.item()

            # Calculate activation entropy
            entropy = calculate_activation_entropy(activations)
            for layer, ent in entropy.items():
                if layer not in history['layer_activation_entropy']:
                    history['layer_activation_entropy'][layer] = []
                history['layer_activation_entropy'][layer].append(ent)
                epoch_activation_entropy[layer] = ent

        # Average train loss
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation pass
        val_loss, val_f1, val_prec, val_rec, cm, report, fpr, tpr, roc_auc = run_eval(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        # Logging
        logging.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Precision: {val_prec:.4f} | "
            f"Recall: {val_rec:.4f} | "
            f"Avg Activation Entropy: {sum(epoch_activation_entropy.values()) / len(epoch_activation_entropy):.4f}"
        )

        # TensorBoard Logging
        if writer:
            # Scalars
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("F1/Val", val_f1, epoch)
            writer.add_scalar("Precision/Val", val_prec, epoch)
            writer.add_scalar("Recall/Val", val_rec, epoch)
            writer.add_scalar("ROC_AUC/Val", roc_auc, epoch)
            writer.add_scalar("Activation Entropy/Train", sum(epoch_activation_entropy.values()) / len(epoch_activation_entropy), epoch)

            # Confusion Matrix
            fig_cm = plot_confusion_matrix(cm, classes=['0', '1'])
            writer.add_figure("Confusion Matrix/Val", fig_cm, epoch)
            plt.close(fig_cm)

            # ROC Curve
            fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
            writer.add_figure("ROC Curve/Val", fig_roc, epoch)
            plt.close(fig_roc)

            # Classification Report as text
            report_text = classification_report_to_text(report)
            writer.add_text("Classification Report/Val", report_text, epoch)

        # Update learning rate scheduler
        scheduler.step(val_f1)
        lr_adjustments = scheduler.num_bad_epochs  # Track LR adjustments
        history['lr_adjustments'] = lr_adjustments

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")

            # Save classification report at best epoch
            history['classification_report'] = classification_report_to_text(report)

            # Save confusion matrix and ROC curve images
            confusion_matrix_path = os.path.join(run_dir, f"confusion_matrix_epoch_{epoch}.png")
            fig_cm.savefig(confusion_matrix_path)
            plt.close(fig_cm)

            roc_curve_path = os.path.join(run_dir, f"roc_curve_epoch_{epoch}.png")
            fig_roc.savefig(roc_curve_path)
            plt.close(fig_roc)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in F1 for {epochs_no_improve} epoch(s).")

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

        # Reset and inject noise based on gradient norms
        grad_norms = calculate_gradient_norms(model)
        reset_params = reset_and_inject_noise(model, grad_norms, threshold=gradient_threshold, noise_level=noise_level)
        history['reset_params'].append(reset_params)
        if reset_params:
            logging.info(f"Epoch {epoch}: Reset and injected noise into parameters: {reset_params}")

        # Check entropy to trigger Surge-Collapse
        avg_entropy = sum(epoch_activation_entropy.values()) / len(epoch_activation_entropy)
        if avg_entropy < entropy_threshold:
            logging.info(f"Epoch {epoch}: Activation entropy ({avg_entropy:.4f}) below threshold ({entropy_threshold}) → Trigger Surge-Collapse")
            # Trigger Surge-Collapse: Reset and inject noise
            reset_params = reset_and_inject_noise(model, grad_norms, threshold=gradient_threshold, noise_level=noise_level)
            if reset_params:
                history['reset_params'][-1].extend(reset_params)
                logging.info(f"Triggered Surge-Collapse: Reset and injected noise into parameters: {reset_params}")

    # Load the best model before returning history
    if best_epoch > 0:
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"Best model loaded from {best_model_path}")

    return history


def classification_report_to_text(report):
    """
    Convert classification report dictionary to a formatted string.
    """
    report_str = "Classification Report\n\n"
    for class_label, metrics in report.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            report_str += f"Class {class_label}:\n"
            report_str += f"  Precision: {metrics['precision']:.4f}\n"
            report_str += f"  Recall:    {metrics['recall']:.4f}\n"
            report_str += f"  F1-Score:  {metrics['f1-score']:.4f}\n\n"
    # Add macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        metrics = report.get(avg_type, {})
        report_str += f"{avg_type.capitalize()}:\n"
        report_str += f"  Precision: {metrics.get('precision', 0):.4f}\n"
        report_str += f"  Recall:    {metrics.get('recall', 0):.4f}\n"
        report_str += f"  F1-Score:  {metrics.get('f1-score', 0):.4f}\n\n"
    # Add overall accuracy
    accuracy = report.get('accuracy', 0)
    report_str += f"Overall Accuracy: {accuracy:.4f}\n"
    return report_str
