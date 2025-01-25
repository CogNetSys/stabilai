import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.surge_collapse_net import SurgeCollapseNet
from models.orthogonal_grad import OrthogonalGrad
from utils.metrics import calculate_entropy, calculate_metrics
from utils.data_loader import collapse_weights, reexpand_weights

def train_model(
    model, train_loader, val_loader, optimizer, criterion, 
    num_epochs=10, collapse_interval=100, surge_interval=200, 
    collapse_sparsity=0.5, surge_recovery=0.1, device='cpu'
):
    """
    Train the model with Surge-Collapse dynamics.
    """
    model.to(device)
    best_f1 = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_entropy = 0.0

        for step, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_entropy = calculate_entropy(targets)
            total_entropy += batch_entropy

            # Surge-Collapse Dynamics
            if (step + 1) % collapse_interval == 0:
                collapse_weights(model, sparsity=collapse_sparsity)
            if (step + 1) % surge_interval == 0:
                reexpand_weights(model, recovery_rate=surge_recovery)

        avg_loss = running_loss / len(train_loader)
        avg_entropy = total_entropy / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:,1]
                preds = torch.argmax(outputs, dim=1)

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        metrics = calculate_metrics(all_labels, all_preds, all_probs)

        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Entropy={avg_entropy:.4f}, Val Loss={avg_val_loss:.4f}, F1={metrics['f1']:.4f}")

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Best model saved.")

    print("Training Complete.")
