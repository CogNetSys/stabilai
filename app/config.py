# app/config.py

# Configuration file for StabilAI project

import argparse

def get_config():
    parser = argparse.ArgumentParser(description="StabilAI Configuration")
    
    # Data parameters
    parser.add_argument('--train_size', type=int, default=4000, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--ood_size', type=int, default=1000, help='Number of OOD samples')
    parser.add_argument('--input_dim', type=int, default=128, help='Input feature dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['base', 'fastgrok', 'noise'], default='base', help='Type of model to train')
    parser.add_argument('--use_gat', action='store_true', help='Use GAT in the model')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    
    # Gradient Filtering and Surge-Collapse
    parser.add_argument('--use_grokfast', action='store_true', help='Enable Grokfast gradient filtering')
    parser.add_argument('--grokfast_type', type=str, choices=['ema', 'ma'], default='ema', help='Type of Grokfast filtering')
    parser.add_argument('--alpha', type=float, default=0.98, help='EMA momentum')
    parser.add_argument('--lamb', type=float, default=2.0, help='Amplification factor')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for MA')
    parser.add_argument('--filter_type', type=str, choices=['mean', 'sum'], default='mean', help='Filter type for MA')
    parser.add_argument('--warmup', action='store_true', help='Enable warmup for MA')
    parser.add_argument('--gradient_threshold', type=float, default=1e-3, help='Threshold for stale gradients')
    parser.add_argument('--noise_level', type=float, default=1e-3, help='Noise level for injected noise')
    parser.add_argument('--dataset_noise_level', type=float, default=0.0, help='Dataset noise injection level')
    parser.add_argument('--entropy_threshold', type=float, default=1.5, help='Entropy threshold for Surge-Collapse')
    
    # Other parameters
    parser.add_argument('--run_dir', type=str, default='runs', help='Directory to save runs and models')
    
    args = parser.parse_args()
    return vars(args)  # Return as a dictionary
