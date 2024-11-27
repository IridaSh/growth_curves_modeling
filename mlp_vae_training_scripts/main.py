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
import argparse
from datetime import datetime

def main(model_choice):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Hyperparameters
    batch_size = 32
    latent_dim = 10
    latent_channel = 16
    lr = 1e-3            # Learning rate
    min_lr = 4e-6        # Minimum learning rate
    epochs = 1000
    gamma = 0.98
    weight_decay = 1e-5
    hidden_size = 128
    warmup_epochs = 8
    patience = 30
    percentage = 0.2
    
   # Validate model choice
    valid_models = ['VAE', 'DeepCNNVAE', 'ResidualCNNVAE']
    if model_choice not in valid_models:
        raise ValueError(f"Invalid model choice '{model_choice}'. Valid options are: {valid_models}")
    
    # File paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (
        'VAE' if model_choice == 'vae' else
        'DeepCNNVAE' if model_choice == 'deep_cnn_vae' else
        'ResidualCNNVAE'
    )
    
    # Define file paths after defining timestamp and model_name
    data_file = 'data/simulated/curves10k.npz'         # Adjust the path as needed
    parameter_file = 'data/simulated/parameters.npy'   # Adjust the path as needed
    vae_model_path = 'train_vae_output/ResidualCNNVAE_LD10_LC16_LR0.001_TS20241126_215733/models/best_model.pt'  # Adjust the path as needed
    output_dir = f'train_vae_mlp_output/{model_name}_LD{latent_dim}_LC{latent_channel}_LR{lr}_TS{timestamp}'
    plots_dir = os.path.join(output_dir, 'plots')
    predictions_save_path = os.path.join(output_dir, 'NN-estimated-parameters.npy')
    
    # Create necessary directories
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'models'))
    create_directory(plots_dir)
    
    # Load and normalize data
    normalized_data, normalized_parameters, input_size, output_size, scaler_data, parameter_scale, parameter_names = normalize_data(
        data_file, parameter_file
    )
    
    # Get indices for train/test split
    indices = np.arange(len(normalized_data))
    
    # Split data into training and testing sets
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
    
    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize models
    mlp_model = MLP(latent_dim, hidden_size, output_size).to(device)
    # Initialize models based on user choice
    if model_choice == 'VAE':
        vae_model = VAE(latent_dim, latent_channel, input_size).to(device)
    elif model_choice == 'DeepCNNVAE':
        vae_model = DeepCNNVAE(latent_dim, latent_channel, input_size).to(device)
    elif model_choice == 'ResidualCNNVAE':
        vae_model = ResidualCNNVAE(latent_dim, latent_channel, input_size).to(device)
    
    logging.info(f'{model_name} model instantiated.')

    # Load the trained VAE model
    try:
        vae_model.load_state_dict(torch.load(vae_model_path))
        logging.info(f'{model_name} model loaded from {vae_model_path}')
    except Exception as e:
        logging.error(f'Error loading {model_name} model from {vae_model_path}: {e}')
        raise
    
    # Freeze VAE model parameters
    for param in vae_model.parameters():
        param.requires_grad = False
    logging.info(f'{model_name} model parameters frozen.')
    
    
    # Create an instance of the combined model
    combined_model = CombinedModel(vae_model, mlp_model).to(device)
    logging.info('Combined model instantiated.')
    
    # Count parameters
    count_parameters(combined_model)
    
    # Loss function and optimizer for training the MLP model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize learning rate schedulers
    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0

    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Initialize early stopping parameters
    best_test_loss = np.inf  # Best test loss so far
    epochs_no_improve = 0  # Counter for epochs since the test loss last improved
    
    # Training loop
    train_loss_values = []
    test_loss_values = []
    
    for epoch in range(epochs):
        train_loss = train(combined_model, train_loader, optimizer, criterion, device)
        test_loss = test(combined_model, test_loader, criterion, device)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)
    
        # Clamp minimum learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
        current_lr = optimizer.param_groups[0]['lr']
    
        # Logging
        interval = 2 if epoch < 10 else 40
        if (epoch + 1) % interval == 0:
            logging.info(f'Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Test Loss: {test_loss:.7f} | Lr: {current_lr:.8f}')
    
        # Update learning rate
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()
    
        # Check for early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0  # Reset the counter
            # Optionally, save the best model here
        else:
            epochs_no_improve += 1  # Increment the counter
    
        if epochs_no_improve >= patience:
            logging.info('Early stopping!')
            break  # Exit the loop
    
    logging.info('Finished Training')
    
    # Plotting the loss values and saving the figure
    loss_plot_path = os.path.join(plots_dir, 'training_testing_loss.png')
    plot_loss_curves(train_loss_values, test_loss_values, loss_plot_path)
    
    # Calculate the number of samples to retrieve
    num_train_samples = int(len(train_dataset) * percentage)
    num_test_samples = int(len(test_dataset) * percentage)
    
    # Get the tensor data without labels
    subset_train_input = train_dataset[:num_train_samples][0]
    subset_test_input = test_dataset[:num_test_samples][0]
    
    with torch.no_grad():
        output_train = combined_model(subset_train_input.to(device))
        output_test = combined_model(subset_test_input.to(device))
    
    # Denormalize the predicted parameters
    subset_train_data = y_train[:num_train_samples].cpu().numpy() * parameter_scale
    subset_test_data = y_test[:num_test_samples].cpu().numpy() * parameter_scale
    output_train = output_train.cpu().numpy() * parameter_scale
    output_test = output_test.cpu().numpy() * parameter_scale
    
    # Plot parameter predictions
    plot_parameter_predictions(
        subset_train_data, output_train, 
        subset_test_data, output_test, 
        parameter_names, 
        plots_dir
    )
    
    # Combine the predicted values (from training and test sets)
    combined_predictions = np.zeros((len(normalized_parameters), output_size))
    
    with torch.no_grad():
        # Full predictions on train and test data
        full_output_train = combined_model(X_train.to(device)).cpu().numpy() * parameter_scale
        full_output_test = combined_model(X_test.to(device)).cpu().numpy() * parameter_scale
    
    # Place the predictions in the combined_predictions array
    combined_predictions[train_indices] = full_output_train
    combined_predictions[test_indices] = full_output_test
    
    # Save the combined predicted parameter values in original order
    save_predictions(combined_predictions, predictions_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VAE model with MLP for parameter prediction.")
    parser.add_argument(
        '--model', 
        type=str, 
        default='VAE', 
        help="Choose the model type: 'VAE', 'DeepCNNVAE', or 'ResidualCNNVAE'. Default is 'VAE'."
    )
    args = parser.parse_args()
    
    # Call the main function with the selected model type
    main(args.model)