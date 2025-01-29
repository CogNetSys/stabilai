from pydantic import BaseModel, Field, validator
import yaml
import logging

class Config(BaseModel):
    """
    Configuration schema for training models, validated with Pydantic.
    """
    train_size: int = Field(4000, description="Number of training samples")
    val_size: int = Field(1000, description="Number of validation samples")
    ood_size: int = Field(1000, description="Number of out-of-distribution samples")
    input_dim: int = Field(128, description="Dimension of input features")
    batch_size: int = Field(64, description="Batch size for training and evaluation")
    model_type: str = Field(
        "base",
        description="Type of model to train",
        pattern="^(base|fastgrok|noise)$"
    )
    num_epochs: int = Field(50, description="Number of training epochs")
    learning_rate: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(1e-5, description="L2 weight decay for optimizer")
    run_dir: str = Field("runs", description="Directory to save logs and models")

    # 🔹 **New Field: Batch Normalization**
    use_batch_norm: bool = Field(True, description="Enable batch normalization in model")
    
    # 🔹 **New Field: Dropout**
    use_dropout: bool = Field(False, description="Enable dropout in model")

    # 🔹 **New Field: Early Stopping Patience**
    early_stopping_patience: int = Field(10, description="Number of epochs with no improvement after which training will be stopped")

    # Advanced training options
    use_grokfast: bool = Field(False, description="Enable Grokfast gradient filtering")
    grokfast_type: str = Field(
        "ema",
        description="Type of Grokfast gradient filter",
        pattern="^(ema|ma)$"
    )
    alpha: float = Field(0.98, description="Momentum for EMA Grokfast filter")
    lamb: float = Field(2.0, description="Amplification factor for Grokfast filter")
    window_size: int = Field(100, description="Window size for MA filter")
    filter_type: str = Field(
        "mean",
        description="Filter type for MA",
        pattern="^(mean|sum)$"
    )
    warmup: bool = Field(True, description="Enable warmup for Grokfast filtering")
    
    # Surge-Collapse & Entropy Based Training
    gradient_threshold: float = Field(0.001, description="Threshold for detecting stale gradients")
    noise_level: float = Field(0.001, description="Noise level injected into stale weights")
    dataset_noise_level: float = Field(0.0, description="Noise level injected into dataset features")
    entropy_threshold: float = Field(1.5, description="Threshold for triggering Surge-Collapse")

    # Optional Parameters for Models
    hidden_size: int = Field(256, description="Number of hidden units in the model")
    output_size: int = Field(2, description="Number of output classes")
    activation_func: str = Field('relu', description="Activation function to use")
    debug: bool = Field(False, description="Enable debug mode")

    @validator('grokfast_type')
    def validate_grokfast_type(cls, v):
        if v not in ('ema', 'ma'):
            raise ValueError("grokfast_type must be either 'ema' or 'ma'")
        return v

    @validator('filter_type')
    def validate_filter_type(cls, v):
        if v not in ('mean', 'sum'):
            raise ValueError("filter_type must be either 'mean' or 'sum'")
        return v

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ('base', 'fastgrok', 'noise'):
            raise ValueError("model_type must be 'base', 'fastgrok', or 'noise'")
        return v


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load configuration from a YAML file and validate using Pydantic.
    """
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:  # Handle empty YAML file
            raise ValueError(f"Configuration file {config_path} is empty!")

        return Config(**config_dict)

    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def save_config(config: Config, config_path: str = "config/config.yaml"):
    """
    Save configuration to a YAML file.
    """
    try:
        with open(config_path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
