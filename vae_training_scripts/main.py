# vae_training_scripts/main.py

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import os
import argparse

from data_loading import load_simulated_data, load_isolates_data
from training import train_epoch, evaluate, count_parameters
from utils import setup_logging, create_output_dir
from plotting import plot_loss, plot_reconstructions, plot_sample_trajectories

# Import your VAE models
from models.shallow_cnn_vae import VAE
from models.deep_cnn_vae import DeepCNNVAE
from models.residual_cnn_vae import ResidualCNNVAE
from models.transformer_vae import TransformerVAE



def main(model_type='ResidualCNNVAE', distribution_type='truncnorm'):
    # Setup logging
    logger = setup_logging()
    logger.info(f"Model type received: {model_type}")
    logger.info(f"Distribution type received: {distribution_type}")

    # Hyperparameters
    batch_size = 32
    latent_dim = 10
    latent_channel = 16
    alpha = 0.5e-4
    lr = 1e-3            # lr=1e-3 for fresh training
    min_lr = 4e-6        # min_lr = 5e-6 for fresh training
    epochs = 1000 if model_type != 'TransformerVAE' else 300  # Transformer runs fewer epochs
    gamma = 0.98
    weight_decay = 1e-5

    # Set data directory based on distribution type
    data_directory = f'data/simulated_{distribution_type}/' if distribution_type == 'uniform' else 'data/simulated/'
    logger.info(f"Using data directory: {data_directory}")

    # Create output directory with distribution information
    output_dir = create_output_dir(
        base_output_dir='train_vae_output',
        model_type=model_type,
        latent_dim=latent_dim,
        latent_channel=latent_channel,
        lr=lr,
        distribution_type=distribution_type
    )

    # Additionally, set up a subdirectory for models
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Load and process simulated data
    logger.info(f"Loading simulated data from {data_directory}")
    all_n_values = load_simulated_data(data_directory, logger)

    # Load and process isolates data
    isolates_data_path = 'data/isolates/OD_311_isolates.npz'
    logger.info(f"Loading isolates data from {isolates_data_path}")
    reshaped_od_values = load_isolates_data(isolates_data_path, logger)

    # Combine the data
    all_data = np.vstack([all_n_values, reshaped_od_values])
    np.random.shuffle(all_data)
    logger.info(f"Combined data shape after shuffling: {all_data.shape}")

    # Prepare data
    if model_type == 'TransformerVAE':
        data = torch.tensor(all_data).float().unsqueeze(1)  # [batch, 1, seq_length]
        seq_length = data.shape[2]
    else:
        data = all_data  
        seq_length = data.shape[1]
        data = torch.tensor(data).float().unsqueeze(1)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.warning("CUDA is not available. Check your CUDA installation and NVIDIA drivers.")

    # Split the data
    train_data, test_data, _, _ = train_test_split(
        data, range(data.shape[0]), test_size=0.1, random_state=42
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    # Initialize model based on the selected architecture
    if model_type == 'VAE':
        model = VAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    elif model_type == 'DeepCNNVAE':
        model = DeepCNNVAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    elif model_type == 'ResidualCNNVAE':
        model = ResidualCNNVAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    elif model_type == 'TransformerVAE':
        # Example: create the TransformerVAE with your desired hyperparams
        model = TransformerVAE(latent_dim=latent_dim,
                               seq_length=seq_length,
                               nhead=4,
                               num_layers=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"The model {model_type} is currently being trained")
    model = model.to(device)

    num_params = count_parameters(model)
    logger.info(f'The model has {num_params:,} parameters')

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate schedulers
    warmup_epochs = 10
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training Loop Variables
    train_loss_values = []
    test_loss_values = []
    best_test_loss = np.inf
    best_state_dict = None
    epochs_no_improve = 0
    patience = 30

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, alpha, device)
        # Evaluate on test set
        test_loss = evaluate(model, test_loader, criterion, device)

        # Record losses
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        # Update learning rate
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()

        # Clamp minimum learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = min_lr
            current_lr = min_lr

        # Logging (less frequent logging after first 10 epochs)
        interval = 2 if epoch < 10 else 40
        if (epoch + 1) % interval == 0:
            logger.info(f'Epoch: {epoch + 1} | '
                        f'Train Loss: {train_loss:.7f}, '
                        f'Test Loss: {test_loss:.7f}, '
                        f'Lr: {current_lr:.8f}')

        # Early Stopping Check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            # Update best_state_dict
            best_state_dict = model.state_dict()
        else:
            epochs_no_improve += 1

        # Trigger Early Stopping
        if epochs_no_improve >= patience:
            logger.info('Early stopping triggered!')
            break

    logger.info("Training completed.")

    # Save the Best Model After Training Ends
    if best_state_dict is not None:
        best_model_path = os.path.join(models_dir, 'best_model.pt')
        torch.save(best_state_dict, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    else:
        logger.warning("No improvement during training. Best model not saved.")

    # Plot loss curves
    plot_loss(train_loss_values, test_loss_values, output_dir)

    # Retrieve a subset of data for reconstruction plots
    percentage = 0.2
    num_train_samples = int(len(train_data) * percentage)
    num_test_samples = int(len(test_data) * percentage)

    subset_train_data = train_data[:num_train_samples].to(device)
    subset_test_data = test_data[:num_test_samples].to(device)

    # Plot reconstructions
    plot_reconstructions(model, subset_train_data, subset_test_data, output_dir, device)

    # Plot sample trajectories
    plot_sample_trajectories(model, subset_train_data, subset_test_data, output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE Models')
    parser.add_argument('--model', type=str, default='VAE',
                        choices=['VAE', 'DeepCNNVAE', 'ResidualCNNVAE', 'TransformerVAE'],
                        help='Specify which VAE architecture to use')
    parser.add_argument('--distribution', type=str, default='truncnorm',
                        choices=['truncnorm', 'uniform'],
                        help='Specify which distribution type to use (truncnorm or uniform)')
    args = parser.parse_args()

    main(model_type=args.model, distribution_type=args.distribution)