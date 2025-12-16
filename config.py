"""
Configuration File
==================
Centralized configuration for the improved recommendation system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # CLIP Encoder
    clip_model_name: str = 'openai/clip-vit-base-patch32'
    freeze_encoders: bool = True
    
    # Sequential User Model
    user_model_dim: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    max_seq_len: int = 50
    dropout: float = 0.1
    
    # Item Embeddings
    use_item_mlp: bool = True
    item_mlp_hidden: int = 256


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Loss
    loss_type: str = 'bpr'  # 'bpr' or 'infonce'
    infonce_temperature: float = 1.0
    
    # Optimization
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 10
    negative_samples: int = 4
    
    # Validation
    validate_every: int = 1
    early_stopping_patience: Optional[int] = None


@dataclass
class DataConfig:
    """Data configuration."""
    # Filtering
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    
    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    k_values: list = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20]


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


# Default configuration
default_config = Config()

