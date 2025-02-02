from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import argparse
import logging
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path

from models.shallow_cnn_vae import VAE
from models.deep_cnn_vae import DeepCNNVAE
from models.residual_cnn_vae import ResidualCNNVAE
from training import calculate_mse
from data_loading import load_simulated_data, load_isolates_data


def create_output_dir(base_dir, model_type, latent_dim, latent_channel):
    """Create a unique output directory based on model configuration and timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{model_type}_LD{latent_dim}_LC{latent_channel}_TS{timestamp}"
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logger(output_dir):
    """Set up logging configuration with a descriptive log file name."""
    log_file = output_dir / f"{output_dir.name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_results(results, output_dir):
    """Save evaluation results to a JSON file with a descriptive name."""
    result_file = output_dir / f"{output_dir.name}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return result_file


def normalize_data(data, logger):
    """Normalize data using MinMaxScaler to [0, 1] range."""
    logger.info("Normalizing data to range [0, 1] using MinMaxScaler.")
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


def load_data(data_directory, isolates_path, logger):
    """Load, normalize, and prepare the dataset."""
    logger.info("Loading simulated data from %s", data_directory)
    all_n_values = load_simulated_data(data_directory, logger)
    
    logger.info("Loading isolates data from %s", isolates_path)
    reshaped_od_values = load_isolates_data(isolates_path, logger)
    
    # Combine data
    all_data = np.vstack([all_n_values, reshaped_od_values])
    
    # Shuffle and convert to torch tensor (matching main.py normalization)
    np.random.seed(42)
    np.random.shuffle(all_data)
    data = torch.tensor(all_data).float().unsqueeze(1)  # [batch, 1, seq_length]
    
    logger.info(f"Combined data shape: {data.shape}")
    logger.info(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    
    return data, None  # Return None instead of scaler since we're not using MinMaxScaler


def get_model(model_type, latent_dim, latent_channel, seq_length, logger):
    """Initialize the specified model."""
    logger.info("Initializing %s model", model_type)
    model_classes = {
        'VAE': VAE,
        'DeepCNNVAE': DeepCNNVAE,
        'ResidualCNNVAE': ResidualCNNVAE
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](
        latent_dim=latent_dim,
        latent_channel=latent_channel,
        seq_length=seq_length
    )


def evaluate_saved_model(args, logger):
    """Evaluate a saved model with the specified parameters."""
    # Load and normalize data
    data, _ = load_data(args.data_directory, args.isolates_path, logger)
    
    # Split the data
    train_data, test_data, _, _ = train_test_split(
        data, range(data.shape[0]), test_size=0.1, random_state=42
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    
    # Initialize model
    model = get_model(
        args.model_type,
        args.latent_dim,
        args.latent_channel,
        data.shape[2],
        logger
    )
    
    # Load model weights
    logger.info("Loading model weights from %s", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Calculate MSE with more detailed logging
    logger.info("Calculating reconstruction MSE")
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Test data range: [{test_data.min():.4f}, {test_data.max():.4f}]")
    
    mse = calculate_mse(model, test_loader, device)
    
    # Log reconstructions for a small batch
    with torch.no_grad():
        sample_batch = next(iter(test_loader)).to(device)
        reconstruction, _, _ = model(sample_batch)
        sample_mse = torch.mean((reconstruction - sample_batch) ** 2).item()
        logger.info(f"Sample batch MSE: {sample_mse:.7f}")
        logger.info(f"Sample input range: [{sample_batch.min():.4f}, {sample_batch.max():.4f}]")
        logger.info(f"Sample reconstruction range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")
    
    logger.info(f"Overall reconstruction MSE: {mse:.7f}")
    
    # Prepare and save results
    results = {
        'model_type': args.model_type,
        'model_path': str(args.model_path),
        'mse': float(mse),
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'latent_dim': args.latent_dim,
            'latent_channel': args.latent_channel,
            'batch_size': args.batch_size
        }
    }
    
    result_file = save_results(results, args.output_dir)
    logger.info("Results saved to %s", result_file)
    
    return mse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained VAE model')
    parser.add_argument('--model-type', type=str, default='ResidualCNNVAE',
                      choices=['VAE', 'DeepCNNVAE', 'ResidualCNNVAE'],
                      help='Type of VAE model to evaluate')
    parser.add_argument('--model-path', type=Path, required=True,
                      help='Path to the saved model weights')
    parser.add_argument('--data-directory', type=Path,
                      default='data/simulated/',
                      help='Directory containing simulated data')
    parser.add_argument('--isolates-path', type=Path,
                      default='data/isolates/OD_311_isolates.npz',
                      help='Path to isolates data file')
    parser.add_argument('--latent-dim', type=int, default=10,
                      help='Dimension of latent space')
    parser.add_argument('--latent-channel', type=int, default=16,
                      help='Number of latent channels')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create output directory with descriptive name
    base_output_dir = "mse_evaluation_results"
    args.output_dir = create_output_dir(
        base_output_dir,
        args.model_type,
        args.latent_dim,
        args.latent_channel,
    )
    
    # Set up logger
    logger = setup_logger(args.output_dir)
    
    try:
        mse = evaluate_saved_model(args, logger)
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        raise