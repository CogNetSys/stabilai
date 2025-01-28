# hyperparameter_tuning/tuner.py

import os
import itertools
import json
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from app.models.surge_collapse_net import SurgeCollapseNet
from app.training import train_fastgrokkingrush, run_eval, plot_metrics
from app.utils.optimizer import OrthogonalGradientOptimizer
from app.utils.metrics import run_eval
from app.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_metrics

# tuner.py

def hyperparameter_tuning(
    train_loader,
    val_loader,
    ood_loader,
    hyperparams_grid,
    device,
    results_dir="hyperparameter_results",
    epochs=50,
    early_stopping_patience=8
):
    """
    Perform hyperparameter tuning by training models across different hyperparameter configurations.
    
    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        ood_loader (DataLoader): OOD data loader.
        hyperparams_grid (dict): Dictionary specifying lists of hyperparameters to tune.
        device (str): Device to train on ('cuda' or 'cpu').
        results_dir (str): Directory to save results.
        epochs (int): Number of training epochs.
        early_stopping_patience (int): Early stopping patience.
    
    Returns:
        None
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate all combinations of hyperparameters
    keys = list(hyperparams_grid.keys())
    values = list(hyperparams_grid.values())
    combinations = list(itertools.product(*values))
    
    logging.info(f"Starting hyperparameter tuning with {len(combinations)} configurations...")
    
    all_results = []
    
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        logging.info(f"Configuration {idx}/{len(combinations)}: {params}")
        
        # Define unique run name based on hyperparameters
        run_name = f"run_{idx}_noiseW{params['noise_level']}_noiseD{params['dataset_noise_level']}_thresh{params['gradient_threshold']}"
        run_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize TensorBoard writer for this run
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
        
        try:
            # Initialize model
            model = SurgeCollapseNet(
                input_size=128, 
                gat_hidden_dim=64, 
                gat_out_dim=64, 
                gat_heads=4, 
                hidden_size=256, 
                output_size=2, 
                use_gat=False,  # Adjust if GAT is to be included
                debug=False
            ).to(device)
            
            # Initialize optimizer
            base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            optimizer = OrthogonalGradientOptimizer(base_optimizer)
            
            # Define loss criterion
            criterion = StableMaxCrossEntropy()
            
            # Train the model with Surge-Collapse and Dataset Noise Injection
            history = train_fastgrokkingrush(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                writer=writer,
                use_grokfast=True,
                grokfast_type="ema",
                alpha=0.98,
                lamb=2.0,
                window_size=100,
                filter_type="mean",
                warmup=True,
                gradient_threshold=params['gradient_threshold'],
                noise_level=params['noise_level'],
                dataset_noise_level=params['dataset_noise_level'],  # Pass dataset_noise_level
                entropy_threshold=1.5,
                run_dir=run_dir
            )
            
            # Evaluate on OOD data
            ood_loss, ood_f1, ood_prec, ood_rec, ood_cm, ood_report, ood_fpr, ood_tpr, ood_auc = run_eval(
                model, ood_loader, criterion, device
            )
            
            # Check if OOD F1 is not NaN
            if np.isnan(ood_f1):
                logging.warning(f"Configuration {run_name} resulted in NaN OOD F1 Score. Skipping this run.")
                continue  # Skip adding this run to results
            
            # Save history to JSON
            history_path = os.path.join(run_dir, "history.json")
            with open(history_path, "w") as f:
                json.dump(history, f)
            
            # Save OOD metrics
            ood_metrics = {
                "ood_loss": ood_loss,
                "ood_f1": ood_f1,
                "ood_precision": ood_prec,
                "ood_recall": ood_rec,
                "ood_auc": ood_auc
            }
            ood_path = os.path.join(run_dir, "ood_metrics.json")
            with open(ood_path, "w") as f:
                json.dump(ood_metrics, f)
            
            # Append to all_results
            result = {
                "run": run_name,
                "hyperparameters": params,
                "history": history,
                "ood_metrics": ood_metrics
            }
            all_results.append(result)
            
            # Optionally, save the model
            model_path = os.path.join(run_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
        
        except Exception as e:
            logging.error(f"Error in configuration {run_name}: {e}")
            continue  # Skip to the next configuration
        
        finally:
            writer.close()

    # Save all_results to a JSON file
    all_results_path = os.path.join(results_dir, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(all_results, f)
    
    logging.info("Hyperparameter tuning completed. Results saved.")
    
    # **Final Results Summary**
    if all_results:
        # Convert to DataFrame for analysis
        df_results = pd.json_normalize(all_results)
        
        # Extract relevant metrics
        df_metrics = df_results[['run', 'hyperparameters.noise_level', 'hyperparameters.dataset_noise_level',
                                 'hyperparameters.gradient_threshold', 'ood_metrics.ood_f1',
                                 'ood_metrics.ood_precision', 'ood_metrics.ood_recall',
                                 'ood_metrics.ood_auc']]
        
        # Rename columns for clarity
        df_metrics.rename(columns={
            'hyperparameters.noise_level': 'Weight Noise Level',
            'hyperparameters.dataset_noise_level': 'Dataset Noise Level',
            'hyperparameters.gradient_threshold': 'Gradient Threshold',
            'ood_metrics.ood_f1': 'OOD F1 Score',
            'ood_metrics.ood_precision': 'OOD Precision',
            'ood_metrics.ood_recall': 'OOD Recall',
            'ood_metrics.ood_auc': 'OOD AUC'
        }, inplace=True)
        
        # Identify the best configuration based on OOD F1 Score
        best_run = df_metrics.loc[df_metrics['OOD F1 Score'].idxmax()]
        logging.info("======================================")
        logging.info("Final Results Summary")
        logging.info("======================================")
        logging.info(f"Best Configuration: {best_run['run']}")
        logging.info(f" - Weight Noise Level: {best_run['Weight Noise Level']}")
        logging.info(f" - Dataset Noise Level: {best_run['Dataset Noise Level']}")
        logging.info(f" - Gradient Threshold: {best_run['Gradient Threshold']}")
        logging.info(f" - OOD F1 Score: {best_run['OOD F1 Score']}")
        logging.info(f" - OOD Precision: {best_run['OOD Precision']}")
        logging.info(f" - OOD Recall: {best_run['OOD Recall']}")
        logging.info(f" - OOD AUC: {best_run['OOD AUC']}")
        logging.info("======================================")
    else:
        logging.warning("No valid results to summarize.")

def visualize_hyperparameter_effects(all_results, results_dir="hyperparameter_results"):
    """
    Create visualizations to analyze the impact of different hyperparameters.
    
    Args:
        all_results (list): List of result dictionaries.
        results_dir (str): Directory to save visualizations.
    
    Returns:
        None
    """
    if not all_results:
        logging.error("No results to visualize.")
        return
    
    df = pd.json_normalize(all_results)
    
    # Extract relevant columns
    required_columns = [
        'hyperparameters.noise_level', 
        'hyperparameters.dataset_noise_level', 
        'hyperparameters.gradient_threshold', 
        'ood_metrics.ood_f1',
        'ood_metrics.ood_precision', 
        'ood_metrics.ood_recall', 
        'ood_metrics.ood_auc'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in results. Filling with NaN.")
            df[col] = np.nan
    
    df = df[required_columns]
    
    # Rename columns for clarity
    df.rename(columns={
        'hyperparameters.noise_level': 'weight_noise_level',
        'hyperparameters.dataset_noise_level': 'dataset_noise_level',
        'hyperparameters.gradient_threshold': 'gradient_threshold',
        'ood_metrics.ood_f1': 'ood_f1',
        'ood_metrics.ood_precision': 'ood_precision',
        'ood_metrics.ood_recall': 'ood_recall',
        'ood_metrics.ood_auc': 'ood_auc'
    }, inplace=True)
    
    # Drop runs with NaN OOD F1
    df_clean = df.dropna(subset=['ood_f1'])
    
    if df_clean.empty:
        logging.error("All runs have NaN 'ood_f1' scores. Cannot perform comparison.")
        return
    
    # Identify top configurations based on OOD F1 Score
    top_configs = df_clean.sort_values(by='ood_f1', ascending=False).head(5)
    logging.info("Top 5 Hyperparameter Configurations based on OOD F1 Score:")
    logging.info(top_configs)
    
    # Save the comparison table
    comparison_path = os.path.join(results_dir, "hyperparameter_comparison.csv")
    df_clean.to_csv(comparison_path, index=False)
    logging.info(f"Hyperparameter comparison table saved to {comparison_path}")
    
    # Initialize the plot
    plt.figure(figsize=(18, 12))
    
    # Heatmap for OOD F1 Score vs Weight Noise and Gradient Threshold
    pivot_f1 = df_clean.pivot_table(index='weight_noise_level', columns='gradient_threshold', values='ood_f1')
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("OOD F1 Score Heatmap")
    plt.xlabel("Gradient Threshold")
    plt.ylabel("Weight Noise Level")
    
    # Heatmap for OOD AUC
    pivot_auc = df_clean.pivot_table(index='weight_noise_level', columns='gradient_threshold', values='ood_auc')
    plt.subplot(2, 2, 2)
    if pivot_auc.empty or pivot_auc.isnull().all().all():
        logging.warning("Unable to generate heatmap: No valid data for OOD AUC.")
        plt.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    else:
        sns.heatmap(pivot_auc, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("OOD AUC Heatmap")
        plt.xlabel("Gradient Threshold")
        plt.ylabel("Weight Noise Level")
    
    # Scatter plot for Dataset Noise Level vs OOD F1 Score
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=df_clean, 
        x='dataset_noise_level', 
        y='ood_f1', 
        hue='weight_noise_level', 
        style='gradient_threshold', 
        s=100, 
        palette='viridis'
    )
    plt.title("Dataset Noise Level vs OOD F1 Score")
    plt.xlabel("Dataset Noise Level")
    plt.ylabel("OOD F1 Score")
    plt.legend(title='Weight Noise & Gradient Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Scatter plot for Weight Noise Level vs OOD AUC
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        data=df_clean, 
        x='weight_noise_level', 
        y='ood_auc', 
        hue='dataset_noise_level', 
        style='gradient_threshold', 
        s=100, 
        palette='magma'
    )
    plt.title("Weight Noise Level vs OOD AUC")
    plt.xlabel("Weight Noise Level")
    plt.ylabel("OOD AUC")
    plt.legend(title='Dataset Noise & Gradient Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hyperparameter_visualizations.png"))
    plt.show()

def generate_summary_report(all_results, report_path="hyperparameter_results/summary_report.md"):
    """
    Generate a markdown report summarizing the hyperparameter tuning results.
    
    Args:
        all_results (list): List of result dictionaries.
        report_path (str): Path to save the summary report.
    
    Returns:
        None
    """
    if not all_results:
        logging.error("No results to generate the summary report.")
        return
    
    df = pd.json_normalize(all_results)
    
    # Extract relevant columns
    required_columns = [
        'run', 
        'hyperparameters.noise_level', 
        'hyperparameters.dataset_noise_level', 
        'hyperparameters.gradient_threshold', 
        'ood_metrics.ood_f1',
        'ood_metrics.ood_precision', 
        'ood_metrics.ood_recall', 
        'ood_metrics.ood_auc'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Missing column '{col}' in results. Cannot generate summary report.")
            return
    
    df = df[required_columns]
    
    # Rename columns for clarity
    df.rename(columns={
        'hyperparameters.noise_level': 'Weight Noise Level',
        'hyperparameters.dataset_noise_level': 'Dataset Noise Level',
        'hyperparameters.gradient_threshold': 'Gradient Threshold',
        'ood_metrics.ood_f1': 'OOD F1 Score',
        'ood_metrics.ood_precision': 'OOD Precision',
        'ood_metrics.ood_recall': 'OOD Recall',
        'ood_metrics.ood_auc': 'OOD AUC'
    }, inplace=True)
    
    # Drop runs with NaN OOD F1
    df_clean = df.dropna(subset=['OOD F1 Score'])
    
    if df_clean.empty:
        logging.error("All runs have NaN 'OOD F1 Score'. Cannot generate summary report.")
        return
    
    # Identify top configuration
    top_config = df_clean.sort_values(by='OOD F1 Score', ascending=False).iloc[0]
    
    # Write to markdown
    with open(report_path, "w") as f:
        f.write("# Hyperparameter Tuning Summary Report\n\n")
        f.write("## Top Performing Configuration\n\n")
        f.write(top_config.to_frame().to_markdown())
        f.write("\n\n## All Configurations Performance\n\n")
        f.write(df_clean.to_markdown(index=False))
        f.write("\n\n## Insights and Findings\n\n")
        f.write(f"- **Optimal Weight Noise Level:** {top_config['Weight Noise Level']:.2f}\n")
        f.write(f"- **Optimal Dataset Noise Level:** {top_config['Dataset Noise Level']:.2f}\n")
        f.write(f"- **Optimal Gradient Threshold:** {top_config['Gradient Threshold']:.4f}\n")
        f.write(f"- **Best OOD F1 Score:** {top_config['OOD F1 Score']:.4f}\n")
        f.write("\n\n## Recommendations\n\n")
        f.write("Based on the tuning results, the optimal hyperparameter settings are as follows:\n")
        f.write(f"- **Weight Noise Level:** {top_config['Weight Noise Level']}\n")
        f.write(f"- **Dataset Noise Level:** {top_config['Dataset Noise Level']}\n")
        f.write(f"- **Gradient Threshold:** {top_config['Gradient Threshold']}\n")
        f.write("\nThese settings yield the highest OOD F1 Score, indicating improved generalization and robustness against out-of-distribution data.")
    
    logging.info(f"Summary report generated at {report_path}")

def compare_hyperparameters(all_results, results_dir="hyperparameter_results"):
    """
    Compare different hyperparameter configurations and identify the best ones.
    
    Args:
        all_results (list): List of result dictionaries.
        results_dir (str): Directory to save comparison table.
    
    Returns:
        None
    """
    if not all_results:
        logging.error("No results to compare.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.json_normalize(all_results)
    
    # Extract relevant columns
    required_columns = [
        'run', 
        'hyperparameters.noise_level', 
        'hyperparameters.dataset_noise_level', 
        'hyperparameters.gradient_threshold', 
        'ood_metrics.ood_f1',
        'ood_metrics.ood_precision', 
        'ood_metrics.ood_recall', 
        'ood_metrics.ood_auc'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in results. Filling with NaN.")
            df[col] = np.nan
    
    df = df[required_columns]
    
    # Rename columns for clarity
    df.rename(columns={
        'hyperparameters.noise_level': 'weight_noise_level',
        'hyperparameters.dataset_noise_level': 'dataset_noise_level',
        'hyperparameters.gradient_threshold': 'gradient_threshold',
        'ood_metrics.ood_f1': 'ood_f1',
        'ood_metrics.ood_precision': 'ood_precision',
        'ood_metrics.ood_recall': 'ood_recall',
        'ood_metrics.ood_auc': 'ood_auc'
    }, inplace=True)
    
    # Drop runs with NaN OOD F1
    df_clean = df.dropna(subset=['ood_f1'])
    
    if df_clean.empty:
        logging.error("All runs have NaN 'ood_f1' scores. Cannot perform comparison.")
        return
    
    # Identify top configurations based on OOD F1 Score
    top_configs = df_clean.sort_values(by='ood_f1', ascending=False).head(5)
    logging.info("Top 5 Hyperparameter Configurations based on OOD F1 Score:")
    logging.info(top_configs)
    
    # Save the comparison table
    comparison_path = os.path.join(results_dir, "hyperparameter_comparison.csv")
    df_clean.to_csv(comparison_path, index=False)
    logging.info(f"Hyperparameter comparison table saved to {comparison_path}")

def generate_comprehensive_report(all_results, results_dir="hyperparameter_results"):
    """
    Generate a comprehensive report comparing different training approaches.
    """
    # Load results
    if not all_results:
        logging.error("No results to generate reports.")
        return
    
    # Compare hyperparameters
    compare_hyperparameters(all_results, results_dir=results_dir)
    
    # Visualize effects
    visualize_hyperparameter_effects(all_results, results_dir=results_dir)
    
    # Generate summary report
    summary_report_path = os.path.join(results_dir, "summary_report.md")
    generate_summary_report(all_results, report_path=summary_report_path)

def main(config_path='app/config.py'):
    # Load configuration
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg = config.get_config()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    
    # Set device
    device = cfg['device'] if 'device' in cfg else ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create DataLoaders
    train_loader, val_loader, ood_loader = get_data_loaders(
        train_size=cfg['train_size'],
        val_size=cfg['val_size'],
        ood_size=cfg['ood_size'],
        input_dim=cfg['input_dim'],
        batch_size=cfg['batch_size']
    )
    
    # Initialize model based on model_type
    model_type = cfg['model_type']
    if model_type == 'base':
        model = BaseModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat']
        )
    elif model_type == 'fastgrok':
        model = FastGrokModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat']
        )
    elif model_type == 'noise':
        model = NoiseModel(
            input_size=cfg['input_dim'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            use_gat=cfg['use_gat'],
            noise_level=cfg['noise_level']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Initialize optimizer and wrap with OrthogonalGradientOptimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    optimizer = OrthogonalGradientOptimizer(base_optimizer)
    
    # Define loss function
    if model_type == 'noise':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = StableMaxCrossEntropy()
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='max', factor=0.5, patience=5)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(cfg['run_dir'], 'training_logs'))
    
    # Train the model
    history = train_fastgrokkingrush(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg['num_epochs'],
        early_stopping_patience=cfg['early_stopping_patience'],
        writer=writer,
        use_grokfast=cfg.get('use_grokfast', False),
        grokfast_type=cfg.get('grokfast_type', 'ema'),
        alpha=cfg.get('alpha', 0.98),
        lamb=cfg.get('lamb', 2.0),
        window_size=cfg.get('window_size', 100),
        filter_type=cfg.get('filter_type', 'mean'),
        warmup=cfg.get('warmup', True),
        gradient_threshold=cfg.get('gradient_threshold', 1e-3),
        noise_level=cfg.get('noise_level', 1e-3),
        dataset_noise_level=cfg.get('dataset_noise_level', 0.0),
        entropy_threshold=cfg.get('entropy_threshold', 1.5),
        run_dir=cfg.get('run_dir', '.')
    )
    
    writer.close()
    logging.info("Training completed.")
