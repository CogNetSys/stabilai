# /training/evaluate_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import create_synthetic_data_loader  # Updated import
from models.surge_collapse_net import SurgeCollapseNet  # Assuming you have this model
from torch.optim import Adam

# Hyperparameters
NUM_SAMPLES = 1000
BATCH_SIZE = 64
INPUT_SIZE = 128  # Adjust based on your data
OUTPUT_SIZE = 1   # Binary classification (0 or 1)
EPOCHS = 5
LEARNING_RATE = 0.001

# Create synthetic dataloaders
train_loader = create_synthetic_data_loader(NUM_SAMPLES, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE)
test_loader = create_synthetic_data_loader(NUM_SAMPLES, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE)

# Model
model = SurgeCollapseNet(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)  # Adjust the model if needed
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss

# Training loop
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())  # Squeeze to match target size
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = (outputs > 0).long()
        correct_predictions += (predictions.squeeze() == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# Evaluation loop
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())  # Squeeze to match target size
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0).long()
            correct_predictions += (predictions.squeeze() == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# Run training and evaluation
for epoch in range(EPOCHS):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
    print("-" * 50)
