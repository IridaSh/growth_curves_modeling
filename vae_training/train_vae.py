import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for script environments
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import logging

def load_simulated_data(data_directory, logger):
    """Load and process simulated data."""
    # Load data
    data_path = os.path.join(data_directory, 'curves10k.npz')
    logger.info(f"Loading simulated data from {data_path}")
    data_array = np.load(data_path)['curves']
    
    # Define conditions
    conditions = [(0, 0), (10, 0), (10, 10)]
    
    # Prepare an empty numpy array to store all concatenated 'n' values
    num_strains = data_array.shape[0]
    num_conditions = data_array.shape[1]
    seq_length = data_array.shape[3]
    all_n_values = np.empty((num_strains, num_conditions * seq_length))
    
    # Extract and concatenate the 'n' values for all conditions
    for strain_index in range(num_strains):
        concatenated_values = np.concatenate(
            [data_array[strain_index, condition_index, 0, :] for condition_index in range(num_conditions)]
        )
        all_n_values[strain_index, :] = concatenated_values

    logger.info(f"Simulated data shape: {all_n_values.shape}")
    return all_n_values

def load_isolates_data(data_path, logger):
    """Load and process isolates data."""
    # Load data
    logger.info(f"Loading isolates data from {data_path}")
    data_array = np.load(data_path)['arr_0']
    
    # Define conditions
    conditions = [(0, 0), (50, 0), (50, 25)]
    
    num_strains = data_array.shape[0]
    num_conditions = len(conditions)
    num_replicates = data_array.shape[2]
    seq_length = data_array.shape[3]
    
    # Prepare an empty numpy array to store all concatenated OD values
    all_od_values = np.empty((num_strains, num_replicates, num_conditions * seq_length))
    
    # Extract and concatenate the OD values for all conditions of each replicate
    for strain_index in range(num_strains):
        for replicate_index in range(num_replicates):
            concatenated_values = np.concatenate(
                [data_array[strain_index, condition_index, replicate_index, :] for condition_index in range(num_conditions)]
            )
            all_od_values[strain_index, replicate_index, :] = concatenated_values

    logger.info(f"Isolates data shape: {all_od_values.shape}")
    # Reshape to 2D array (num_samples, num_features)
    reshaped_od_values = all_od_values.reshape(-1, all_od_values.shape[2])
    return reshaped_od_values

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

def train(model, dataloader, optimizer, criterion, alpha, device):
    """Training loop for one epoch."""
    model.train()
    running_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        reconstruction, mean, logvar = model(data)
        recon_loss = criterion(reconstruction, data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + alpha * kl_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test(model, dataloader, criterion, device):
    """Validation loop."""
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            reconstruction, _, _ = model(data)
            loss = criterion(reconstruction, data)

            running_loss += loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def get_latent_variables(model, dataloader, device):
    """Retrieve latent variables from the encoder."""
    model.eval()
    all_latent_vars = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, mean, _ = model(data)
            all_latent_vars.append(mean.detach().cpu())
    return torch.cat(all_latent_vars)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(filename='vae_training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Output directory for saving plots and outputs
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and process simulated data
    data_directory = 'data/simulated/'
    all_n_values = load_simulated_data(data_directory, logger)

    # Load and process isolates data
    isolates_data_path = 'data/isolates/OD_311_isolates.npz'
    reshaped_od_values = load_isolates_data(isolates_data_path, logger)

    # Combine the data
    all_data = np.vstack([all_n_values, reshaped_od_values])

    # Shuffle the data
    np.random.shuffle(all_data)
    logger.info(f"Combined data shape after shuffling: {all_data.shape}")

    # Proceed with rest of the code using 'all_data' instead of 'all_n_values'
    data = all_data
    seq_length = data.shape[1]

    # Determine device (enables GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        # Log detailed CUDA device information
        logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        # Log an error if CUDA is not available
        logger.error("CUDA is not available. Check your CUDA installation and NVIDIA drivers.")

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

    data = torch.tensor(data).float().unsqueeze(1)

    # Split the data into train and test sets
    train_data, test_data, train_indices, test_indices = train_test_split(
        data, range(data.shape[0]), test_size=0.1, random_state=42)

    train_data = train_data.clone().detach().float()
    test_data = test_data.clone().detach().float()

    # Prepare DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    # Model creation, loss function, and optimizer
    # Initialize model with seq_length explicitly
    model = VAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    # Load previous model if it exists
    # model.load_state_dict(torch.load('trained/sim_exp_L10/VAE_sim_exp_L10.7.3.pt'))

    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f'The model has {num_params:,} parameters')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    train_loss_values = []
    test_loss_values = []

    # Initialize early stopping parameters
    best_test_loss = np.inf  # Best test loss so far
    epochs_no_improve = 0    # Counter for epochs since the test loss last improved
    patience = 30            # Patience for early stopping

    # Implement a warmup schedule to start from a small learning rate.
    warmup_epochs = 10

    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0

    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, alpha, device)
        test_loss = test(model, test_loader, criterion, device)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        # Clamp minimum learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)
            current_lr = param_group['lr']

        interval = 2 if epoch < 10 else 40
        if (epoch + 1) % interval == 0:
            logger.info(f'Epoch: {epoch + 1} Train Loss: {train_loss_values[epoch]:.7f}, '
                        f'Test Loss: {test_loss_values[epoch]:.7f}, Lr: {current_lr:.8f}')

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
            logger.info('Early stopping!')
            break  # Exit the loop

    logger.info("Training completed.")

    # After training, get the latent variables
    train_latent_vars = get_latent_variables(model, train_loader, device)
    test_latent_vars = get_latent_variables(model, test_loader, device)

    # Save the model
    model_path = "/hpc/group/youlab/is178"
    model_save_path = os.path.join(model_path, 'vae_model.pt')
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Plotting the loss values
    plt.figure(figsize=(6, 3))
    plt.semilogy(train_loss_values, label='Training')
    plt.semilogy(test_loss_values, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved to {loss_plot_path}")

    # Percentage of data to retrieve
    percentage = 0.2
    num_train_samples = int(len(train_data) * percentage)
    num_test_samples = int(len(test_data) * percentage)

    # Index into the data tensors
    subset_train_data = train_data[:num_train_samples].to(device)
    subset_test_data = test_data[:num_test_samples].to(device)

    with torch.no_grad():
        output_train, _, _ = model(subset_train_data)
        output_test, _, _ = model(subset_test_data)

    # Squeeze the output to match the original data dimension
    output_train = output_train.squeeze(1)
    output_test = output_test.squeeze(1)

    output_train = output_train.cpu().numpy()
    output_test = output_test.cpu().numpy()

    subset_train_data = subset_train_data.cpu().numpy().squeeze(1)
    subset_test_data = subset_test_data.cpu().numpy().squeeze(1)

    # Plotting the reconstructed data against the original data
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Training data plot
    axs[0].scatter(subset_train_data.flatten(), output_train.flatten(),
                   s=0.1, color='blue', alpha=0.5)
    axs[0].plot([np.min(subset_train_data), np.max(subset_train_data)],
                [np.min(subset_train_data), np.max(subset_train_data)], 'r')  # y=x line
    axs[0].set_xlim(np.min(subset_train_data), np.max(subset_train_data))
    axs[0].set_ylim(np.min(subset_train_data), np.max(subset_train_data))
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlabel('Original')
    axs[0].set_ylabel('Reconstructed')
    axs[0].set_title('Training')
    r2_train = r2_score(subset_train_data.flatten(), output_train.flatten())
    axs[0].text(0.05, 0.95, f'R^2 = {r2_train:.3f}',
                transform=axs[0].transAxes, verticalalignment='top')

    # Testing data plot
    axs[1].scatter(subset_test_data.flatten(), output_test.flatten(),
                   s=0.1, color='blue', alpha=0.5)
    axs[1].plot([np.min(subset_test_data), np.max(subset_test_data)],
                [np.min(subset_test_data), np.max(subset_test_data)], 'r')  # y=x line
    axs[1].set_xlim(np.min(subset_test_data), np.max(subset_test_data))
    axs[1].set_ylim(np.min(subset_test_data), np.max(subset_test_data))
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlabel('Original')
    axs[1].set_ylabel('Reconstructed')
    axs[1].set_title('Testing')
    r2_test = r2_score(subset_test_data.flatten(), output_test.flatten())
    axs[1].text(0.05, 0.95, f'R^2 = {r2_test:.3f}',
                transform=axs[1].transAxes, verticalalignment='top')

    plt.tight_layout()
    reconstruction_scatter_path = os.path.join(output_dir, 'reconstruction_scatter.png')
    plt.savefig(reconstruction_scatter_path)
    plt.close()
    logger.info(f"Reconstruction scatter plot saved to {reconstruction_scatter_path}")

    # Additional Panels for Sample Data Trajectories
    fig, axs = plt.subplots(2, 5, figsize=(10, 6))

    # Training data trajectories
    for i in range(5):
        axs[0, i].plot(subset_train_data[i], label='Original', color='blue')
        axs[0, i].plot(output_train[i], label='Reconstructed', color='orange')
        axs[0, i].set_title(f'Training {i + 1}')
        # axs[0, i].legend()

    # Testing data trajectories
    for i in range(5):
        axs[1, i].plot(subset_test_data[i], label='Original', color='blue')
        axs[1, i].plot(output_test[i], label='Reconstructed', color='orange')
        axs[1, i].set_title(f'Testing {i + 1}')
        # axs[1, i].legend()

    plt.tight_layout()
    sample_trajectories_path = os.path.join(output_dir, 'sample_trajectories.png')
    plt.savefig(sample_trajectories_path)
    plt.close()
    logger.info(f"Sample trajectories plot saved to {sample_trajectories_path}")