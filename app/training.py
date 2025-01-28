# app/training.py

import torch
import torch.nn as nn
from app.models.surge_collapse_net import SurgeCollapseNet
from app.utils.optimizer import OrthogonalGradientOptimizer
from app.utils.metrics import run_eval
from app.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_metrics
from app.utils.gradfilter import gradfilter_ma, gradfilter_ema
from app.data.data_loader import get_data_loaders, add_gaussian_noise
import logging
import os
from sklearn.metrics import classification_report

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
    and Dataset Noise Injection.

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
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'layer_activation_entropy': {},
        'reset_params': []
    }

    # Initialize Grokfast gradient storage
    if use_grokfast and grokfast_type == 'ema':
        grads = None  # Will be initialized in gradfilter_ema
    elif use_grokfast and grokfast_type == 'ma':
        grads = None  # Will be initialized in gradfilter_ma
    else:
        grads = None  # Not used

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

            # ROC Curve
            fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
            writer.add_figure("ROC Curve/Val", fig_roc, epoch)

            # Classification Report as text
            report_text = classification_report_to_text(report)
            writer.add_text("Classification Report/Val", report_text, epoch)

        # Update learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.base_optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        scheduler.step(val_f1)

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # Save the best model
            os.makedirs(run_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in F1 for {epochs_no_improve} epoch(s).")

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

        # Reset and inject noise based on gradient norms
        grad_norms = calculate_gradient_norms(model)
        reset_params = reset_and_inject_noise(model, grad_norms, gradient_threshold, noise_level)
        history['reset_params'].append(reset_params)
        if reset_params:
            logging.info(f"Epoch {epoch}: Reset and injected noise into parameters: {reset_params}")

        # Check entropy to trigger Surge-Collapse
        avg_entropy = sum(epoch_activation_entropy.values()) / len(epoch_activation_entropy)
        if avg_entropy < entropy_threshold:
            logging.info(f"Epoch {epoch}: Activation entropy ({avg_entropy:.4f}) below threshold ({entropy_threshold}) → Trigger Surge-Collapse")
            # Trigger Surge-Collapse: Reset and inject noise
            reset_params = reset_and_inject_noise(model, grad_norms, gradient_threshold, noise_level)
            if reset_params:
                history['reset_params'][-1].extend(reset_params)
                logging.info(f"Triggered Surge-Collapse: Reset and injected noise into parameters: {reset_params}")

    # Final metrics plot (optional)
    if writer:
        # You can create custom plots here or use external visualization
        pass

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


def main(config_path='app/config.py'):
    # Load configuration
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg = config.get_config()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    
    # Set device
    device = cfg['device'] if 'device' in cfg else ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create DataLoaders
    train_loader, val_loader, ood_loader = get_data_loaders(
        train_size=cfg['train_size'],
        val_size=cfg['val_size'],
        ood_size=cfg['ood_size'],
        input_dim=cfg['input_dim'],
        batch_size=cfg['batch_size']
    )
    
    # Initialize model based on model_type
    model_type = cfg['model_type']
    if model_type == 'base':
        model = BaseModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat']
        )
    elif model_type == 'fastgrok':
        model = FastGrokModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat']
        )
    elif model_type == 'noise':
        model = NoiseModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat'],
            noise_level=cfg['noise_level']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Initialize optimizer and wrap with OrthogonalGradientOptimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    optimizer = OrthogonalGradientOptimizer(base_optimizer)
    
    # Define loss function
    if model_type == 'noise':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = StableMaxCrossEntropy()
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='max', factor=0.5, patience=5)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(cfg['run_dir'], 'training_logs'))
    
    # Train the model
    history = train_fastgrokkingrush(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg['num_epochs'],
        early_stopping_patience=cfg['early_stopping_patience'],
        writer=writer,
        use_grokfast=cfg.get('use_grokfast', False),
        grokfast_type=cfg.get('grokfast_type', 'ema'),
        alpha=cfg.get('alpha', 0.98),
        lamb=cfg.get('lamb', 2.0),
        window_size=cfg.get('window_size', 100),
        filter_type=cfg.get('filter_type', 'mean'),
        warmup=cfg.get('warmup', True),
        gradient_threshold=cfg.get('gradient_threshold', 1e-3),
        noise_level=cfg.get('noise_level', 1e-3),
        dataset_noise_level=cfg.get('dataset_noise_level', 0.0),
        entropy_threshold=cfg.get('entropy_threshold', 1.5),
        run_dir=cfg.get('run_dir', '.')
    )
    
    writer.close()
    logging.info("Training completed.")
