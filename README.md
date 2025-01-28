# StabilAI

Welcome to the StabilAI project! This repository encapsulates the training and evaluation of the SurgeCollapseNet model, a custom neural network tailored for binary classification tasks using PyTorch.

## 📌 About This Project

StabilAI is designed to demonstrate advanced machine learning techniques, including synthetic data generation, custom neural network architectures, hyperparameter tuning, and comprehensive model evaluation. The project emphasizes modularity, configurability, and professional best practices, making it both robust and adaptable for various classification challenges.

## 🔍 Features

- **Synthetic Data Generation**: Create synthetic datasets for binary classification.
- **Custom Neural Network Models**: Implement and toggle between Base Model, FastGrok, and Noise Models.
- **Hyperparameter Tuning**: Centralized configuration for easy hyperparameter adjustments and optimization.
- **Advanced Training Techniques**: Incorporates early stopping, learning rate scheduling, and gradient filtering.
- **Model Evaluation**: Evaluate models using metrics like F1-score, precision, recall, and ROC-AUC.
- **Visualization**: Monitor training progress and metrics using TensorBoard.
- **Documentation**: Comprehensive documentation using MkDocs for easy navigation and understanding.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (`venv`)

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/stabilai.git
    cd stabilai
    ```

2. **Set Up Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**

**Note:** this installs torch 2.5.1 configured for CUDA 12.4. Run 'nvcc --version' to determine your CUDA version. If your CUDA is different then change the number from 124 to 121 if you use 12.1 or 118 if you use 11.8.

    ```bash
    pip install torch torchvision tensorboard scikit-learn seaborn pynvml
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

    ```

4. **Run the Application**

    ```bash
    PYTHONPATH=$(pwd) python3 app/main.py --config app/config.py
    ```

### Configuration

All key parameters are centralized in the `app/config.py` file. This setup allows easy toggling between different models (Base Model, FastGrok, Noise Model) and adjusting hyperparameters without delving into the core codebase.

### Documentation

Generate and view the documentation using MkDocs:

```bash
mkdocs serve
Access the documentation at http://127.0.0.1:8000/.
```

📚 Documentation
Comprehensive documentation is available in the docs/ directory and can be served locally or hosted via GitHub Pages.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 5. `app/__init__.py`

```python
# app/__init__.py
```

# 📈 Centralized Configuration and Model Toggling  
All key parameters are managed in the `app/config.py` file. This setup allows you to toggle between different models and adjust hyperparameters without modifying the core codebase. Here's how to adjust settings:

### Choose Model Type  
In `app/config.py`, set the `--model_type` argument:  

```python
parser.add_argument('--model_type', type=str, choices=['base', 'fastgrok', 'noise'], default='base', help='Type of model to train')
```

### Enable GAT  
To enable Graph Attention Networks (GAT), add the `--use_gat` flag:  

```bash
python app/main.py --config app/config.py --use_gat
```

### Adjust Hyperparameters  
Modify hyperparameters like learning rate, batch size, noise levels, etc., directly in `app/config.py` or pass them as command-line arguments:  

```bash
python app/main.py --config app/config.py --learning_rate 0.001 --batch_size 128
```

### Centralized Hyperparameter Tuning  
All hyperparameters are centralized, allowing easy integration with hyperparameter tuning tools or scripts. Adjust parameters in `app/config.py` and rerun the training or tuning scripts without searching through the codebase.  

# What Next?

1. Add Learning Rate Scheduler:

2. Incorporate learning rate schedulers to adjust the learning rate during training based on validation metrics.

3. Implement Early Stopping with Patience:

4. Enhance the early stopping mechanism to monitor additional metrics or implement more sophisticated stopping criteria.

5. Save and Load Model Checkpoints:

6. Implement functionality to save intermediate model checkpoints and load them for resuming training or inference.