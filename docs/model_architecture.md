# 🌟 Model Architecture

The core of the **Surge-Collapse Training with Entropy Dynamics** project revolves around a simplified **Auto-Regressive Neural Network** designed for sequential or structured data.

---

## 🏗️ **Architecture Overview**

- **🔹 Input Layer**: Receives input data vectors.
- **🔹 Hidden Layer**: 
  - **Type**: Fully Connected (Linear)
  - **Activation**: ReLU (Rectified Linear Unit)
  - **Units**: 256
- **🔹 Output Layer**: Produces the final predictions.

---

## 🛠️ **Model Definition**

Here’s the definition of the model:

```python
import torch
import torch.nn as nn

class AutoRegressiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

---

## 🌟 **Key Features**

1. **⚙️ Simplicity**: The model's straightforward architecture facilitates easy integration and experimentation.
2. **📏 Flexibility**: Can be adapted for various tasks by modifying input and output dimensions.
3. **✨ ReLU Activation**: Promotes sparsity and non-linearity, enhancing the model's expressive power.

---

## 🔄 **Surge-Collapse Training**

The **Surge-Collapse Training** mechanism introduces adaptive weight pruning and re-expansion techniques aimed at optimizing training stability and performance.

---

## 🧩 **Mechanism Breakdown**

### **1. ⚡ Collapse (Weight Pruning)**

- **Objective**: Prune low-magnitude weights to promote sparsity and reduce redundancy.
- **Method**: Set weights below a specified sparsity threshold to zero.

```python
def collapse_weights(model, sparsity=0.5):
    with torch.no_grad():
        for param in model.parameters():
            threshold = torch.quantile(torch.abs(param), sparsity)
            param[param.abs() < threshold] = 0
```

---

### **2. 🌱 Surge (Weight Re-Expansion)**

- **Objective**: Reintroduce pruned weights with controlled noise to prevent dead weights and allow recovery of useful parameters.
- **Method**: Inject random noise scaled by a recovery factor into pruned weights.

```python
def reexpand_weights(model, recovery_rate=0.1):
    with torch.no_grad():
        for param in model.parameters():
            mask = param == 0
            param[mask] = torch.randn(mask.sum(), device=param.device) * recovery_rate
```

---

## 🔄 **Adaptive Surge-Collapse**

The Surge-Collapse process adapts based on **activation entropy levels**, ensuring that the network dynamically responds to its training state.

### 🔹 **High Entropy**:
- **Trigger Collapse** to reduce redundancy.
- **Inject Energy** to maintain training momentum.
- Apply **Entropy Pumps** for information flow.

### 🔹 **Low Entropy**:
- Add **Controlled Noise** to inputs to sustain information diversity.

### 🔹 **Entropy Plateaus**:
- **Initiate Collapse** for regularization to prevent stagnation.

---

## 🔧 **Integration into Training Loop**

The Surge-Collapse dynamics are integrated into the training loop at specified intervals to maintain optimal training conditions.

```python
def train_with_surge_collapse(model, data_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            # Surge-Collapse Dynamics
            if i % 100 == 0:  # Collapse every 100 steps
                collapse_weights(model, sparsity=0.5)
            if i % 200 == 0:  # Surge every 200 steps
                reexpand_weights(model, recovery_rate=0.1)

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
```

---

## ✅ **Benefits**

1. **🔒 Stabilized Training**: Prevents overfitting and underfitting by dynamically adjusting the network's sparsity.
2. **📈 Enhanced Generalization**: Promotes the discovery of robust features through controlled weight adjustments.
3. **⚙️ Flexibility**: Can be tailored to different tasks by modifying sparsity thresholds and recovery rates.

---

## 🏁 **Conclusion**

Surge-Collapse Training offers a robust framework for optimizing neural network training dynamics, ensuring stability and enhancing performance through **adaptive weight management** and **entropy-based strategies**.