# **SurgeCollapseNet: Adaptive Learning for Generalization**

SurgeCollapseNet is a neural network architecture and training methodology inspired by the concept of balancing **"surge"** (active learning of meaningful patterns) and **"collapse"** (reducing redundancy for generalization). This project integrates ideas from **entropy-driven learning**, **numerical stability**, and **pragmatic model design** to create a robust system for binary classification and beyond.

---

## **Key Features**
- A fully connected feedforward neural network designed for binary classification tasks.
- Adaptive training dynamics inspired by the **surge-collapse** philosophy.
- Advanced training techniques including **early stopping**, **learning rate scheduling**, and **entropy-based evaluation**.
- TensorBoard integration for visualization of training and validation metrics.
- Modular, extensible code that encourages experimentation and collaboration.

---

## **Contributors**
### **1. Richard Aragon (Entropy-Driven Dynamics)**
Richard contributed the foundational idea of using **adaptive surge-collapse dynamics** for neural networks:
- Introduced **weight collapse and surge** mechanisms driven by activation entropy.
- Developed an **adaptive entropy thresholding system** to dynamically guide training.
- Inspired the inclusion of entropy tracking for evaluating model confidence and uncertainty.

### **2. Lucas Prieto et al. (The Grokking Paper Authors)**
The concepts of **numerical stability**, **delayed generalization**, and the mathematical underpinnings of "grokking" in neural networks were drawn from the groundbreaking research in the paper:
- Highlighted the importance of **logit stability** and the conditions under which generalization occurs.
- Inspired the integration of numerical stability concepts into the training pipeline.
- Provided theoretical insights that guided the design of the SurgeCollapseNet architecture.

### **3. [Your Name] (Practical Implementation and Training Framework)**
Your contributions focused on transforming theoretical insights into a **scalable, application-driven implementation**:
- Designed the SurgeCollapseNet architecture with **multi-layer perceptrons**, **ReLU activations**, and **adaptive regularization**.
- Integrated **TensorBoard** for real-time visualization of training metrics.
- Added advanced training techniques, including:
  - **Early stopping** to prevent overfitting.
  - **Learning rate scheduling** for smoother convergence.
- Built a modular training pipeline with hooks for entropy measurement, loss tracking, and evaluation.

---

## **Code Structure**
```
📁 SurgeCollapseNet
├── 📁 models
│   ├── surge_collapse_net.py       # Model architecture for SurgeCollapseNet
├── 📁 training
│   ├── train_model.py              # Training loop with early stopping and TensorBoard integration
│   ├── utils.py                    # Helper functions (entropy calculation, metric tracking)
├── 📁 datasets
│   ├── synthetic_data_loader.py    # Code for generating synthetic binary classification datasets
├── 📁 visualizations
│   ├── tensorboard_logs            # TensorBoard logs for training and validation metrics
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## **How to Run the Project**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/SurgeCollapseNet.git
   cd SurgeCollapseNet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   Run the training script with default parameters:
   ```bash
   python training/train_model.py
   ```

4. **Visualize training**:
   Launch TensorBoard to monitor training and validation metrics:
   ```bash
   tensorboard --logdir=visualizations/tensorboard_logs
   ```

5. **Evaluate the model**:
   After training, load the best model and evaluate it on the validation dataset:
   ```bash
   python training/evaluate_model.py
   ```

---

## **Credits**
This project stands on the shoulders of giants:
- **Richard Aragon** for his groundbreaking exploration of entropy-driven adaptive training dynamics.
- **Lucas Prieto et al.** for the inspiring research paper *"Grokking at the Edge of Numerical Stability"*.
- Open-source AI and machine learning communities for tools like PyTorch, TensorBoard, and more.

---

## **License**
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code.

---

## **Contributing**
We welcome contributions! Feel free to submit pull requests or raise issues to help improve SurgeCollapseNet. See our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.