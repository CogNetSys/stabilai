# Surge-Collapse Training with Entropy Dynamics

Welcome to the **Surge-Collapse Training with Entropy Dynamics** project repository. This project introduces innovative training techniques aimed at stabilizing neural network training through adaptive weight pruning and re-expansion, coupled with entropy-based analysis.

---

## **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributions](#contributions)
- [License](#license)

---

## **Overview**

Surge-Collapse Training is an adaptive mechanism that dynamically prunes and re-expands neural network weights based on entropy measurements. This approach aims to enhance training stability, maintain energy balance, and improve generalization across various tasks and architectures.

---

## **Features**

- **Adaptive Weight Pruning (Collapse)**: Removes low-magnitude weights to promote sparsity.
- **Weight Re-Expansion (Surge)**: Reintroduces pruned weights with controlled noise to prevent dead weights.
- **Entropy-Based Analysis**: Monitors activation and target entropy to guide training dynamics.
- **Custom Optimizers**: Includes the ⊥Grad optimizer to prevent Naïve Loss Minimization.
- **StableMax Activation**: A numerically stable Softmax variant to prevent Softmax Collapse.
- **Comprehensive Documentation**: Detailed guides and explanations using MkDocs.

---

## **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

---

Create a Virtual Environment

bash
Copy
python3 -m venv venv
source venv/bin/activate
Install Dependencies

bash
Copy
pip install -r requirements.txt
Note: Ensure that requirements.txt includes all necessary packages, such as torch, mkdocs, mkdocs-material, etc.

Usage
Run Experiments

Navigate to the scripts/ directory and execute the desired training scripts.

bash
Copy
python train.py
View Documentation

Serve the documentation locally using MkDocs.

bash
Copy
mkdocs serve
Access the documentation at http://127.0.0.1:8000/ in your browser.

Build Documentation

Generate static site files for deployment.

bash
Copy
mkdocs build
Documentation
Comprehensive documentation is available using MkDocs with the Material theme. It covers all aspects of the project, including model architecture, training mechanisms, experiments, results, and more.

Access Documentation


## **Credits and Contributions**
This project stands on the shoulders of giants:
- **Richard Aragon** for his groundbreaking exploration of entropy-driven adaptive training dynamics.
- **Lucas Prieto et al.** for the inspiring research paper *"Grokking at the Edge of Numerical Stability"*.
- Open-source AI and machine learning communities for tools like PyTorch, TensorBoard, and more.

---

## **License**
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code.
