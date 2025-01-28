# app/main.py

import argparse
import json
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from app.models import SurgeCollapseNet
from app.training import train_fastgrokkingrush   
from app.data.data_loader import get_data_loaders    
from app.utils.optimizer import OrthogonalGradientOptimizer
from app.utils.losses import StableMaxCrossEntropy  
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set to DEBUG for more verbosity
    handlers=[logging.StreamHandler()]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # Add other arguments for flexibility
    parser.add_argument('--train_size', type=int, default=4000, help='Training dataset size')
    parser.add_argument('--val_size', type=int, default=1000, help='Validation dataset size')
    parser.add_argument('--ood_size', type=int, default=1000, help='OOD dataset size')
    parser.add_argument('--input_dim', type=int, default=128, help='Input feature dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model_type', type=str, choices=['base', 'fastgrok', 'noise'], default='base', help='Type of model to train')
    parser.add_argument('--use_gat', action='store_true', help='Enable Graph Attention Network (GAT)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--run_dir', type=str, default='runs', help='Directory to save logs and models')
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}:")
        print(config)

        # Overwrite command-line arguments with config values
        for key, value in config.items():
            setattr(args, key, value)

    # Print final configuration
    print("Final configuration:")
    print(vars(args))

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data loaders
    train_loader, val_loader, ood_loader = get_data_loaders(
        train_size=args.train_size,
        val_size=args.val_size,
        ood_size=args.ood_size,
        input_dim=args.input_dim,
        batch_size=args.batch_size
    )

    # Initialize the model based on model_type
    if args.model_type == 'base':
        print("Initializing Base Model...")
        model = SurgeCollapseNet(
            input_size=args.input_dim,
            hidden_size=256,
            output_size=2,
            use_batch_norm=True,
            use_dropout=False,
            activation_func='relu',
            debug=False
        )
    elif args.model_type == 'fastgrok':
        print("Initializing FastGrok Model...")
        # You can customize FastGrokModel differently if needed
        model = SurgeCollapseNet(
            input_size=args.input_dim,
            hidden_size=256,
            output_size=2,
            use_batch_norm=True,
            use_dropout=False,
            activation_func='relu',
            debug=False
        )
    elif args.model_type == 'noise':
        print("Initializing Noise Model...")
        # You can customize NoiseModel differently if needed
        model = SurgeCollapseNet(
            input_size=args.input_dim,
            hidden_size=256,
            output_size=2,
            use_batch_norm=True,
            use_dropout=False,
            activation_func='relu',
            debug=False
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(device)

    # Set up optimizer and loss function
    optimizer = OrthogonalGradientOptimizer(
        torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    )
    criterion = StableMaxCrossEntropy()

    # Initialize TensorBoard SummaryWriter
    run_dir = os.path.join(args.run_dir, args.model_type)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))

    # Train the model
    print(f"Training {args.model_type} model...")
    train_fastgrokkingrush(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.num_epochs,
        early_stopping_patience=8,  # You can make this configurable
        writer=writer,
        use_grokfast=args.__dict__.get('use_grokfast', False),
        grokfast_type=args.__dict__.get('grokfast_type', 'ema'),
        alpha=args.__dict__.get('alpha', 0.98),
        lamb=args.__dict__.get('lamb', 2.0),
        window_size=args.__dict__.get('window_size', 100),
        filter_type=args.__dict__.get('filter_type', 'mean'),
        warmup=args.__dict__.get('warmup', True),
        gradient_threshold=args.__dict__.get('gradient_threshold', 0.001),
        noise_level=args.__dict__.get('noise_level', 0.001),
        dataset_noise_level=args.__dict__.get('dataset_noise_level', 0.0),
        entropy_threshold=args.__dict__.get('entropy_threshold', 1.5),
        run_dir=run_dir
    )

    writer.close()
    print(f"Model training completed. Results saved to {run_dir}")

if __name__ == "__main__":
    main()
