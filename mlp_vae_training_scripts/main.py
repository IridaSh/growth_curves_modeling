# main.py

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models.vae_base_mlp import MLP, CombinedModel
from data_loading import normalize_data
from training import train, test, count_parameters
from plotting import plot_loss_curves, plot_parameter_predictions
from utils import create_directory, save_predictions
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from vae_training_scripts.models.shallow_cnn_vae import VAE
from vae_training_scripts.models.deep_cnn_vae import DeepCNNVAE
from vae_training_scripts.models.residual_cnn_vae import ResidualCNNVAE
from vae_training_scripts.models.transformer_vae import TransformerVAE
import argparse
from datetime import datetime

def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Hyperparameters (now with arg-based overrides)
    batch_size = 32
    latent_dim = 10
    latent_channel = 16
    lr = 1e-3
    min_lr = 4e-6
    epochs = 1000
    gamma = 0.98
    weight_decay = 1e-5
    hidden_size = 128
    warmup_epochs = 8
    patience = 30
    percentage = 0.2

    # Validate model choice
    valid_models = ['VAE', 'DeepCNNVAE', 'ResidualCNNVAE', 'TransformerVAE']
    if args.model not in valid_models:
        raise ValueError(f"Invalid model choice '{args.model}'. Valid options are: {valid_models}")
    
    # File paths configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detect distribution type from data file path
    distribution_type = 'uniform' if 'simulated_uniform' in args.data_file else 'truncnorm'
    
    model_name = (
        'VAE' if args.model == 'VAE' else
        'DeepCNNVAE' if args.model == 'DeepCNNVAE' else
        'TransformerVAE' if args.model == 'TransformerVAE' else
        'ResidualCNNVAE'
    )
    
    output_dir = f'train_vae_mlp_output/{model_name}_LD{latent_dim}_LC{latent_channel}_LR{lr}_DIST{distribution_type}_TS{timestamp}'
    
    # Create directory structure
    plots_dir = os.path.join(output_dir, 'plots')
    models_dir = os.path.join(output_dir, 'models')
    predictions_save_path = os.path.join(output_dir, 'NN-estimated-parameters.npy')
    
    create_directory(output_dir)
    create_directory(models_dir)
    create_directory(plots_dir)
    
    # Load and normalize data
    normalized_data, normalized_parameters, input_size, output_size, scaler_data, parameter_scale, parameter_names = normalize_data(
        args.data_file, args.parameter_file
    )
    
    # Split data
    indices = np.arange(len(normalized_data))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        normalized_data, 
        normalized_parameters, 
        indices,
        test_size=0.2, 
        random_state=42
    )
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    # DataLoader setup
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                             batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(TensorDataset(X_test, y_test), 
                            batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model initialization
    mlp_model = MLP(latent_dim, hidden_size, output_size).to(device)
    
    if args.model == 'VAE':
        vae_model = VAE(latent_dim, latent_channel, input_size).to(device)
    elif args.model == 'DeepCNNVAE':
        vae_model = DeepCNNVAE(latent_dim, latent_channel, input_size).to(device)
    elif args.model == 'ResidualCNNVAE':
        vae_model = ResidualCNNVAE(latent_dim, latent_channel, input_size).to(device)
    elif args.model == "TransformerVAE":
        vae_model = TransformerVAE(
            latent_dim=latent_dim, seq_length=input_size, nhead=4,num_layers=3).to(device)    
    logging.info(f'{model_name} model instantiated.')

    # Load and freeze VAE
    try:
        vae_model.load_state_dict(torch.load(args.vae_model_path))
        logging.info(f'Loaded {model_name} from {args.vae_model_path}')
    except Exception as e:
        logging.error(f'Error loading VAE: {e}')
        raise
    
    for param in vae_model.parameters():
        param.requires_grad = False
    
    # Combined model setup
    combined_model = CombinedModel(vae_model, mlp_model).to(device)
    logging.info('Combined model ready.')
    count_parameters(combined_model)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate schedulers
    def warmup_scheduler(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_scheduler)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Training loop
    best_test_loss = np.inf
    epochs_no_improve = 0
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        train_loss = train(combined_model, train_loader, optimizer, criterion, device)
        test_loss = test(combined_model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Learning rate management
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if (epoch + 1) % (2 if epoch < 10 else 40) == 0:
            logging.info(f'Epoch {epoch+1:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | LR: {current_lr:.2e}')
        
        # Update schedulers
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            torch.save(combined_model.state_dict(), os.path.join(models_dir, 'best_combined_model.pt'))
            logging.info(f'New best model saved with test loss: {test_loss:.4f}')
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping after {patience} epochs without improvement')
            break
    
    # Final model save
    torch.save(combined_model.state_dict(), os.path.join(models_dir, 'final_combined_model.pt'))
    logging.info(f'Training complete. Models saved to {models_dir}')
    
    # Post-processing
    plot_loss_curves(train_losses, test_losses, os.path.join(plots_dir, 'training_curves.png'))
    
    # Prediction and plotting
    sample_size = int(len(train_loader.dataset) * percentage)
    with torch.no_grad():
        train_pred = combined_model(X_train[:sample_size].to(device)).cpu().numpy() * parameter_scale
        test_pred = combined_model(X_test[:sample_size].to(device)).cpu().numpy() * parameter_scale
    
    plot_parameter_predictions(
        y_train[:sample_size].numpy() * parameter_scale,
        train_pred,
        y_test[:sample_size].numpy() * parameter_scale,
        test_pred,
        parameter_names,
        plots_dir
    )
    
    # Save full predictions
    with torch.no_grad():
        full_preds = np.zeros_like(normalized_parameters) * parameter_scale
        full_preds[train_indices] = combined_model(X_train.to(device)).cpu().numpy() * parameter_scale
        full_preds[test_indices] = combined_model(X_test.to(device)).cpu().numpy() * parameter_scale
    
    save_predictions(full_preds, predictions_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE+MLP Parameter Estimation")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='VAE',
                       choices=['VAE', 'DeepCNNVAE', 'ResidualCNNVAE', 'TransformerVAE'],
                       help='Choice of VAE architecture')
    
    # Data paths
    parser.add_argument('--data_file', type=str, 
                       default='data/simulated/curves10k.npz',
                       help='Path to input light curve data')
    parser.add_argument('--parameter_file', type=str,
                       default='data/simulated/parameters.npy',
                       help='Path to simulation parameters')
    parser.add_argument('--vae_model_path', type=str,
                       default='train_vae_output/VAE_LD10_LC16_LR0.001_TS20241126_002111/models/best_model.pt',
                       help='Path to pretrained VAE model')
    
    args = parser.parse_args()
    main(args)