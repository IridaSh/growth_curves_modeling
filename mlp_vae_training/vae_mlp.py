import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

def load_data(file_path):
    """Load data from an npz file."""
    return np.load(file_path)['curves']

def concatenate_time_series(data_array):
    """
    Concatenate the 'n' values for each sample across conditions.

    Parameters:
    - data_array: 4D numpy array containing time-series data.

    Returns:
    - all_n_values: 2D numpy array where each row represents a concatenated time series for a strain.
    """
    num_samples = data_array.shape[0]
    num_conditions = data_array.shape[1]
    num_time_points = data_array.shape[3]

    # Initialize an empty array to store all concatenated 'n' values
    all_n_values = np.empty((num_samples, num_conditions * num_time_points))

    # Concatenate time series for each strain across conditions
    for strain_index in range(num_samples):
        all_n_values[strain_index, :] = np.concatenate(
            [data_array[strain_index, condition_index, 0, :] for condition_index in range(num_conditions)]
        )

    return all_n_values

def round_parameters(parameters, decimals=3):
    """
    Rounds the given numpy array to a specified number of decimal places.

    Parameters:
    - parameters: numpy array, the array of parameters to be rounded.
    - decimals: int, the number of decimal places to round to (default is 3).

    Returns:
    - numpy array with values rounded to the specified decimal places.
    """
    return np.round(parameters, decimals=decimals)

def normalize_data(data_file, parameter_file):
    """Normalize the input data and parameters."""
    data_array = load_data(data_file)

    # Concatenate time series
    data = concatenate_time_series(data_array)
    scaler_data = MinMaxScaler()
    normalized_data = scaler_data.fit_transform(data)

    # Load and round parameters
    parameters = np.load(parameter_file)
    logging.info(f'Parameters shape: {parameters.shape}')
    logging.info(f'Max parameter values before rounding: {np.max(parameters, axis=0)}')

    # Round the parameters
    parameters = round_parameters(parameters, decimals=3)
    logging.info(f'Max parameter values after rounding: {np.max(parameters, axis=0)}')
    parameter_names = ['alpha','Ks','Nm', 'theta', 'kappab', 'phimax', 'gamma', 'betamin', 'db', 'c']

    # Dynamically calculate parameter_scale as the rounded-up maximum values
    parameter_scale = np.ceil(np.max(parameters, axis=0) * 1000) / 1000  # Round up to the nearest 0.001
    logging.info(f'Parameter scale (rounded-up max values): {parameter_scale}')

    # Normalize parameters by dividing by parameter_scale
    normalized_parameters = parameters / parameter_scale

    input_size = normalized_data.shape[1]
    output_size = normalized_parameters.shape[1]

    return normalized_data, normalized_parameters, input_size, output_size, scaler_data, parameter_scale, parameter_names

# Define models
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = torch.clamp(out, min=0, max=1)  # Force the output to be within 0 & 1.
        return out

class CNNEncoder(nn.Module):
    """CNN Encoder for VAE."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, latent_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(latent_channel * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(latent_channel * seq_length, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class CNNDecoder(nn.Module):
    """CNN Decoder for VAE."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(latent_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.latent_channel, self.seq_length)
        x = self.decoder(x)
        return F.relu(x)

class VAE(nn.Module):
    """Variational Autoencoder using CNN architecture."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(latent_dim, latent_channel, seq_length)
        self.decoder = CNNDecoder(latent_dim, latent_channel, seq_length)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CombinedModel(nn.Module):
    def __init__(self, vae_model, mlp_model):
        super(CombinedModel, self).__init__()
        self.mlp_model = mlp_model
        self.vae_model = vae_model

    def forward(self, x):
        x = x.unsqueeze(1) 
        with torch.no_grad():
            mean, logvar = self.vae_model.encoder(x)
            latent_variables = self.vae_model.reparameterize(mean, logvar)
        predicted_parameters = self.mlp_model(latent_variables)
        return predicted_parameters

def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0
    for inputs, targets in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        predicted_parameters = model(inputs)
        predicted_parameters = predicted_parameters.view_as(targets)
        
        # Compute loss
        loss = criterion(predicted_parameters, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted_parameters = model(inputs)
            predicted_parameters = predicted_parameters.view_as(targets)
            
            # Compute loss
            loss = criterion(predicted_parameters, targets)
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def denormalize_parameters(normalized_parameters, parameter_scale):
    return normalized_parameters * parameter_scale

# Main function
if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    latent_dim = 10
    latent_channel = 16
    alpha = 0.5e-4
    lr = 1e-3            # lr=1e-3 for fresh training
    min_lr = 4e-6        # min_lr = 5e-6 for fresh training
    epochs = 1000
    gamma = 0.98
    weight_decay = 1e-5
    hidden_size = 128

    # Load and normalize data
    data_file = 'data/simulated/curves10k.npz'  # Adjust the path as needed
    parameter_file = 'data/simulated/parameters.npy'  # Adjust the path as needed
    normalized_data, normalized_parameters, input_size, output_size, scaler_data, parameter_scale, parameter_names = normalize_data(data_file, parameter_file)

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
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Create an instance of the MLP model
    mlp_model = MLP(latent_dim, hidden_size, output_size).to(device)
    # Create an instance of the VAE model
    seq_length = input_size  # Assuming input_size is the sequence length
    vae_model = VAE(latent_dim, latent_channel, seq_length).to(device)

    # Load the trained VAE model
    vae_model.load_state_dict(torch.load('/hpc/group/youlab/is178/vae_model.pt'))

    # Freeze VAE model parameters
    for param in vae_model.parameters():
        param.requires_grad = False

    # Create an instance of the combined model
    combined_model = CombinedModel(vae_model, mlp_model).to(device)

    # Loss function and optimizer for training the MLP model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize learning rate schedulers
    warmup_epochs = 8
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
    patience = 30  # Patience for early stopping

    # Training loop
    train_loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        train_loss = train(combined_model, train_loader, optimizer, criterion)
        test_loss = test(combined_model, test_loader, criterion)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        # Clamp minimum learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
        current_lr = optimizer.param_groups[0]['lr']

        interval = 2 if epoch < 10 else 40
        if (epoch + 1) % interval == 0:
            logging.info('Epoch: {} Train Loss: {:.7f}, Test Loss: {:.7f}, Lr: {:.8f}'.format(epoch + 1, train_loss, test_loss, current_lr))

        # Update learning rate
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()

        # Check for early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        if epochs_no_improve == patience:
            logging.info('Early stopping!')
            break  # Exit the loop

    logging.info('Finished Training')

    # Create directory to save plots
    plots_dir = 'vae_mlp_plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Plotting the loss values and saving the figure
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_loss_values, label='Training Loss')
    plt.semilogy(test_loss_values, label='Testing Loss')
    plt.title('Training & Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_testing_loss.png'))
    plt.close()

    # Percentage of data to retrieve
    percentage = 0.2

    # Calculate the number of samples to retrieve
    num_train_samples = int(len(train_data) * percentage)
    num_test_samples = int(len(test_data) * percentage)

    # Get the tensor data without labels
    subset_train_input = train_data[:num_train_samples][0]
    subset_test_input = test_data[:num_test_samples][0]

    with torch.no_grad():
        output_train = combined_model(subset_train_input.to(device))
        output_test = combined_model(subset_test_input.to(device))

    # Denormalize the predicted parameters
    subset_train_data = denormalize_parameters(train_data[:num_train_samples][1].cpu().numpy(), parameter_scale)
    subset_test_data = denormalize_parameters(test_data[:num_test_samples][1].cpu().numpy(), parameter_scale)
    output_train = denormalize_parameters(output_train.cpu().numpy(), parameter_scale)
    output_test = denormalize_parameters(output_test.cpu().numpy(), parameter_scale)

    # Loop over the columns (parameters)
    for col in range(output_train.shape[1]):
        # Create a new figure for each parameter
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Adjusted figsize for better visibility

        # Training data plot
        axs[0].scatter(subset_train_data[:, col], output_train[:, col], s=5, color='blue', alpha=0.5)
        min_val = min(np.min(subset_train_data[:, col]), np.min(output_train[:, col]))
        max_val = max(np.max(subset_train_data[:, col]), np.max(output_train[:, col]))
        axs[0].plot([min_val, max_val], [min_val, max_val], 'r')  # y=x line
        axs[0].set_xlim(min_val, max_val)
        axs[0].set_ylim(min_val, max_val)
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].set_xlabel('Original')
        axs[0].set_ylabel('Predicted')
        axs[0].set_title(f'Train: {parameter_names[col]}')
        r2_train = r2_score(subset_train_data[:, col], output_train[:, col])
        axs[0].text(0.05, 0.95, f'R² = {r2_train:.3f}', transform=axs[0].transAxes, verticalalignment='top')

        # Testing data plot
        axs[1].scatter(subset_test_data[:, col], output_test[:, col], s=5, color='green', alpha=0.5)
        min_val = min(np.min(subset_test_data[:, col]), np.min(output_test[:, col]))
        max_val = max(np.max(subset_test_data[:, col]), np.max(output_test[:, col]))
        axs[1].plot([min_val, max_val], [min_val, max_val], 'r')  # y=x line
        axs[1].set_xlim(min_val, max_val)
        axs[1].set_ylim(min_val, max_val)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_xlabel('Original')
        axs[1].set_ylabel('Predicted')
        axs[1].set_title(f'Test: {parameter_names[col]}')
        r2_test = r2_score(subset_test_data[:, col], output_test[:, col])
        axs[1].text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=axs[1].transAxes, verticalalignment='top')

        plt.tight_layout()
        # Save the figure
        plot_filename = f'{parameter_names[col]}_prediction.png'
        plt.savefig(os.path.join(plots_dir, plot_filename))
        plt.close()

    # Combine the predicted values (from training and test sets)
    # Create a placeholder for the combined predictions in the original order
    combined_predictions = np.zeros((len(normalized_parameters), output_size))

    # Place the predictions in their original places
    with torch.no_grad():
        # Full predictions on train and test data
        full_output_train = combined_model(X_train.to(device)).cpu().numpy()
        full_output_test = combined_model(X_test.to(device)).cpu().numpy()

    # Denormalize the predicted parameters
    full_output_train = denormalize_parameters(full_output_train, parameter_scale)
    full_output_test = denormalize_parameters(full_output_test, parameter_scale)

    # Place the predictions in the combined_predictions array
    combined_predictions[train_indices] = full_output_train
    combined_predictions[test_indices] = full_output_test

    # Save the combined predicted parameter values in original order
    np.save('NN-estimated-parameters.npy', combined_predictions)