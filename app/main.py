import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from app.config import load_config, Config
from app.models import BaseModel, FastGrokModel, NoiseModel, SurgeCollapseNet  # Import all models
from app.training import train_fastgrokkingrush
from app.data.data_loader import get_data_loaders
from app.utils.optimizer import OrthogonalGradientOptimizer
from app.utils.losses import StableMaxCrossEntropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    return parser.parse_args()


def setup_logging(run_dir: str):
    """
    Set up logging to both console and a file.

    Args:
        run_dir (str): Directory where logs will be saved.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up run directory
    run_dir = os.path.join(config.run_dir, config.model_type)
    os.makedirs(run_dir, exist_ok=True)

    # Set up logging
    setup_logging(run_dir)
    logging.info("======================================")
    logging.info("Training Started")
    logging.info("======================================")
    logging.info(f"Configuration: {config}")

    # Initialize TensorBoard SummaryWriter
    tensorboard_logs = os.path.join(run_dir, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tensorboard_logs)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Prepare data loaders
    train_loader, val_loader, ood_loader = get_data_loaders(
        train_size=config.train_size,
        val_size=config.val_size,
        ood_size=config.ood_size,
        input_dim=config.input_dim,
        batch_size=config.batch_size
    )

    # Initialize the model based on model_type
    logging.info(f"Initializing {config.model_type.capitalize()} Model...")
    if config.model_type == 'base':
        model = BaseModel(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            use_batch_norm=config.use_batch_norm,
            use_dropout=config.use_dropout,
            activation_func=config.activation_func,
            debug=config.debug
        )
    elif config.model_type == 'fastgrok':
        model = FastGrokModel(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            use_gat=config.use_grokfast,  # Assuming 'use_grokfast' relates to GAT usage
            use_dropout=config.use_dropout,
            activation_func=config.activation_func,
            debug=config.debug
        )
    elif config.model_type == 'noise':
        model = NoiseModel(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            use_gat=config.use_grokfast,  # Assuming 'use_grokfast' relates to GAT usage
            use_dropout=config.use_dropout,
            noise_level=config.noise_level,
            activation_func=config.activation_func,
            debug=config.debug
        )
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    model.to(device)
    logging.info(f"Model initialized: {model}")

    # Set up optimizer and loss function
    optimizer = OrthogonalGradientOptimizer(
        torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    )
    criterion = StableMaxCrossEntropy()

    # Train the model
    logging.info(f"Starting training for {config.num_epochs} epochs...")
    history = train_fastgrokkingrush(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience,
        writer=writer,
        use_grokfast=config.use_grokfast,
        grokfast_type=config.grokfast_type,
        alpha=config.alpha,
        lamb=config.lamb,
        window_size=config.window_size,
        filter_type=config.filter_type,
        warmup=config.warmup,
        gradient_threshold=config.gradient_threshold,
        noise_level=config.noise_level,
        dataset_noise_level=config.dataset_noise_level,
        entropy_threshold=config.entropy_threshold,
        run_dir=run_dir
    )

    # Save final model
    model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Final model saved to {model_path}")

    # Generate report using Jinja2
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.md.j2')
    report_content = template.render(history=history, config=config)

    report_path = os.path.join(run_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)

    logging.info(f"Training report generated at {report_path}")

    writer.close()
    logging.info("======================================")
    logging.info("Training Completed")
    logging.info("======================================")


if __name__ == "__main__":
    main()
