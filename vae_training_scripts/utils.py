# utils.py
import logging
import os
from datetime import datetime

def setup_logging(log_file='vae_training.log'):
    """Setup logging configuration."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

def create_output_dir(base_output_dir='output', model_type='VAE', latent_dim=10, latent_channel=16, lr=1e-3):
    """
    Create a unique output directory based on model configuration and timestamp.
    
    Parameters:
    - base_output_dir (str): Base directory for outputs.
    - model_type (str): Type/name of the model architecture.
    - latent_dim (int): Dimension of the latent space.
    - latent_channel (int): Number of latent channels.
    - lr (float): Learning rate.
    
    Returns:
    - output_dir (str): Path to the created output directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{model_type}_LD{latent_dim}_LC{latent_channel}_LR{lr}_TS{timestamp}"
    output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir