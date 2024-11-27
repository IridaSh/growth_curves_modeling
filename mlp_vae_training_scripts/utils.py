# utils.py

import os
import numpy as np
import logging

def create_directory(dir_path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f'Directory created or already exists: {dir_path}')
    except Exception as e:
        logging.error(f'Error creating directory {dir_path}: {e}')
        raise

def save_predictions(combined_predictions, save_path):
    """Save the combined predicted parameter values."""
    try:
        np.save(save_path, combined_predictions)
        logging.info(f'Combined predictions saved to {save_path}')
    except Exception as e:
        logging.error(f'Error saving predictions to {save_path}: {e}')
        raise