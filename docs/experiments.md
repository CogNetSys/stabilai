# 🧪 **Experiments**

This section outlines the experiments conducted to validate the effectiveness of Surge-Collapse Training combined with entropy dynamics.

---

## 📊 **1. Dataset Configurations**

### **a. Dummy Dataset**
- **Purpose**: Simulate training scenarios with controlled inputs and targets.
- **Parameters**:
  - **Input Size**: 128
  - **Output Size**: 128
  - **Number of Samples**: 10,000
  - **Batch Size**: 64
  - **Noise**: Added Gaussian noise to simulate real-world scenarios.

### **b. Modular Arithmetic**
- **Tasks**: Modular addition, multiplication, and subtraction.
- **Modulus**: 113 (a prime number for better structure).
- **Training/Test Split**: Varies depending on experiments (e.g., 40%/60%, 60%/40%, 70%/30%).
- **Challenge**: Tests the network’s ability to generalize on algorithmic reasoning tasks.

### **c. Sparse Parity**
- **Description**: Predict the parity of `k` bits out of a binary vector of length `n`, where `k ≪ n`.
- **Parameters**:
  - **Number of Samples**: 2,000
  - **Train/Test Split**: 50%/50%.
  - **Challenge**: Sparse input requires the model to focus on meaningful features.

### **d. MNIST Subset**
- **Purpose**: Validate Surge-Collapse dynamics on image classification tasks.
- **Parameters**:
  - **Number of Training Samples**: 200 (subsampled from MNIST for controlled experiments).
  - **Test Set**: Full MNIST test set.
- **Challenge**: Tests whether the adaptive dynamics work well on complex, real-world data.

---

## 🏗 **2. Model Configurations**

### **a. Multi-Layer Perceptron (MLP)**
- **Architecture**: 2 hidden layers with 200 units each.
- **Activation**: ReLU.
- **Purpose**: Simple yet expressive model to test generalization dynamics.

### **b. Transformer Model**
- **Architecture**: One-layer transformer for algorithmic tasks.
- **Attention Heads**: 4.
- **Training**: Full batch settings.
- **Purpose**: Evaluate entropy dynamics in sequence-based tasks.

---

## ⚙️ **3. Training Settings**

- **Loss Function**: Cross-Entropy Loss (replaced with StableMax for some experiments).
- **Optimizers**:
  - **Adam**: Baseline optimizer with a learning rate of 0.001 and weight decay of 1e-5.
  - **Orthogonal Gradient Optimizer**: Custom optimizer integrating Adam with orthogonal gradient updates.
- **Epochs**: 50 for most experiments.
- **Early Stopping**: Patience of 10 epochs to prevent overfitting.
- **Entropy-Based Adaptive Mechanisms**: Integrated for weight pruning (collapse) and re-expansion (surge).

---

## 🔧 **4. Intervention Methods**

### **a. StableMax Cross Entropy (StCE) Loss**
- **Objective**: Prevent Softmax Collapse by introducing a numerically stable Softmax variant.
- **Implementation**: Replace standard Softmax with StableMax in the loss calculation.

### **b. ⊥Grad Optimizer**
- **Objective**: Prevent Naïve Loss Minimization by orthogonalizing gradient updates.
- **Implementation**: Modify the optimizer to project gradients orthogonal to the current weight direction.

---

## 📈 **5. Evaluation Metrics**

### **Training Dynamics**
- **Training Loss**: Monitor convergence trends.
- **Validation Loss**: Assess generalization performance.
- **Entropy Measures**:
  - **Activation Entropy**: Captures uncertainty in network activations.
  - **Target Entropy**: Measures diversity in target labels.

### **Classification Metrics**
- **Accuracy**: Standard performance metric.
- **F1 Score**: Balance between precision and recall.
- **Confusion Matrix**: Evaluate per-class performance.
- **ROC Curve and AUC**: Assess discriminative capability.

---

## 🧩 **6. Experiment Execution**

```python
# Example: Training with Surge-Collapse Dynamics
history = train_with_surge_collapse(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=50
)
```

### Key Steps:
1. **Initialize Model and Dataset**: Configure the model (e.g., MLP or Transformer) and dataset (e.g., Modular Arithmetic or MNIST Subset).
2. **Optimizer Selection**: Use either Adam or Orthogonal Gradient Optimizer.
3. **Apply Interventions**: Integrate StableMax Cross Entropy Loss and/or Surge-Collapse Dynamics.
4. **Train and Evaluate**: Track loss, entropy, and metrics across training epochs.

---

## 📊 **7. Summary of Findings**

1. **Stabilized Training**: Surge-Collapse Dynamics effectively mitigated training instabilities.
2. **Enhanced Generalization**: Models demonstrated faster convergence and improved performance on validation sets.
3. **Entropy Dynamics**: Activation entropy provided insights into the network's capacity and stability throughout training.
4. **Intervention Effectiveness**:
   - **StableMax Loss**: Reduced numerical instabilities and improved class-wise balance.
   - **⊥Grad Optimizer**: Prevented gradient alignment issues, fostering faster generalization.

---

## 🔄 **8. Reproducibility**

- **Code Availability**: All experiments are designed to be reproducible. The complete implementation is available in the [GitHub Repository](#).
- **Instructional Guides**: Detailed instructions for setting up datasets, models, and training pipelines are provided.

---

## 🚀 **9. Future Experiments**

1. **Scaling to Larger Datasets**: Evaluate Surge-Collapse Training on more complex and large-scale datasets such as ImageNet or CIFAR-100.
2. **Integration with Different Architectures**: Test the effectiveness across various neural network architectures, such as CNNs and deep transformers.
3. **Hyperparameter Optimization**: Explore optimal settings for sparsity thresholds, recovery rates, and entropy thresholds.
4. **Grokfast Integration**: Investigate the combined effect of Surge-Collapse Training and the Grokfast gradient amplification methods.
