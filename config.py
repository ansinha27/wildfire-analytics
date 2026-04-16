# config.py
#
# Central configuration for the Ferguson Wildfire Analytics
# pipeline.
#
# To switch between local and Azure:
#   - Locally:  set ENVIRONMENT = "local"  (default)
#   - Azure:    set ENVIRONMENT = "azure"
#               and set environment variables:
#               AZURE_STORAGE_CONNECTION_STRING
#               COSMOS_CONNECTION_STRING

import os
from dataclasses import dataclass, field


# Environment Switch

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")


# Data Paths

@dataclass
class DataConfig:
    """
    Paths to raw data files.
    Local:  relative paths to data/ folder
    Azure:  blob names within your container
    """
    train_path:      str = "data/Ferguson_fire_train.npy"
    test_path:       str = "data/Ferguson_fire_test.npy"
    background_path: str = "data/Ferguson_fire_background.npy"
    obs_path:        str = "data/Ferguson_fire_obs.npy"


# Linear Compression (TruncatedSVD)

@dataclass
class CompressionConfig:
    """
    Hyperparameters for TruncatedSVD compression.
    n_components: number of modes to retain
    var_threshold: target cumulative variance (0.95 = 95%)
    batch_size: number of frames to process at once
                keeps memory usage low on large datasets
    """
    n_components:  int   = 114
    var_threshold: float = 0.95
    subset_size:   int   = 500
    batch_size:    int   = 50
    n_iter:        int   = 7


# Nonlinear Compression (UNet Autoencoder)

@dataclass
class AEConfig:
    """
    Hyperparameters for the UNet Autoencoder.

    latent_dim:  bottleneck size — matched to TSVD n_components
                 so comparisons are fair
    batch_size:  32 fills GPU memory efficiently
    lr:          1e-3 initial, halved every 30 epochs
    alpha_mse:   weight on pixel-wise reconstruction loss
    alpha_edge:  weight on Sobel edge loss — preserves
                 sharp fire boundaries
    patience:    early stopping patience in epochs
    """
    latent_dim:  int   = 114
    batch_size:  int   = 32
    n_epochs:    int   = 100
    lr:          float = 1e-3
    step_size:   int   = 30
    gamma:       float = 0.5
    alpha_mse:   float = 1.0
    alpha_edge:  float = 0.1
    patience:    int   = 10
    delta:       float = 1e-5


# Data Assimilation (Kalman Filter)

@dataclass
class AssimilationConfig:
    """
    Kalman Filter configuration.

    beta: observation error scaling factor.
          R = beta * I in latent space.
          Tuned by validation sweep — 0.68 gave best
          MSE in both linear and nonlinear experiments.
          Too small: over-fits noisy observations.
          Too large: ignores valuable observations.
    """
    beta: float = 0.68


# Azure Configuration

@dataclass
class AzureConfig:
    """
    Azure service configuration.
    All values read from environment variables —
    never hardcode credentials.
    """
    storage_connection_str: str = field(
        default_factory=lambda: os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING", ""
        )
    )
    cosmos_connection_str: str = field(
        default_factory=lambda: os.getenv(
            "COSMOS_CONNECTION_STRING", ""
        )
    )
    container_name: str = "ferguson-wildfire"
    cosmos_db:      str = "wildfire-results"
    cosmos_container: str = "pipeline-runs"


# Paths - Models & Outputs

@dataclass
class PathConfig:
    """
    Where models and outputs are saved.
    """
    models_dir:       str = "models"
    outputs_dir:      str = "outputs"
    tsvd_model:       str = "models/tsvd_model.joblib"
    mean_train:       str = "models/mean_train.npy"
    ae_weights:       str = "models/unet_ae.pt"
    ae_checkpoint:    str = "models/unet_ae_checkpoint.pt"
    mean_map:         str = "models/mean_map.npy"
    std_map:          str = "models/std_map.npy"
    ae_latent:        str = "models/unet_latent.npy"
    centered_memmap:  str = "models/train_cent.dat"


# Storage Factory
# Returns correct storage backend based on ENVIRONMENT

def get_storage():
    """
    Factory function.
    Returns LocalStorage or AzureStorage
    depending on ENVIRONMENT variable.

    Usage:
        from config import get_storage
        storage = get_storage()
        data = storage.load_array("data/train.npy")

    To switch to Azure:
        export ENVIRONMENT=azure
        export AZURE_STORAGE_CONNECTION_STRING=...
        export COSMOS_CONNECTION_STRING=...
    """
    if ENVIRONMENT == "azure":
        from src.storage.azure_blob import AzureStorage
        cfg = AzureConfig()
        return AzureStorage(
            container_name=cfg.container_name,
            cosmos_db=cfg.cosmos_db,
            cosmos_container=cfg.cosmos_container
        )
    else:
        from src.storage.local import LocalStorage
        return LocalStorage()


# Instantiate configs 

data_config        = DataConfig()
compression_config = CompressionConfig()
ae_config          = AEConfig()
assimilation_config = AssimilationConfig()
path_config        = PathConfig()
azure_config       = AzureConfig()