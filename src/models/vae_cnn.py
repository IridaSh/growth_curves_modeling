import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convolutional and Transpose Convolutional Blocks with Batch Normalization
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.block(x)

# Enhanced CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, seq_length, latent_channel, latent_dim):
        super(CNNEncoder, self).__init__()
        self.seq_length = seq_length
        self.latent_channel = latent_channel
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1),
            ConvBlock(16, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 1, 1),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, latent_channel, 3, 1, 1)
        )
        self.fc_mean = nn.Linear(latent_channel * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(latent_channel * seq_length, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

# Enhanced CNN Decoder
class CNNDecoder(nn.Module):
    def __init__(self, seq_length, latent_channel, latent_dim):
        super(CNNDecoder, self).__init__()
        self.seq_length = seq_length
        self.latent_channel = latent_channel
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        self.decoder = nn.Sequential(
            ConvTransposeBlock(latent_channel, 128, 3, 1, 1, 0),
            ConvTransposeBlock(128, 64, 3, 1, 1, 0),
            ConvTransposeBlock(64, 32, 3, 1, 1, 0),
            ConvTransposeBlock(32, 16, 3, 1, 1, 0),
            ConvTransposeBlock(16, 3, 3, 1, 1, 0)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.latent_channel, self.seq_length)  # Reshape to (batch, channels, seq_length)
        return torch.sigmoid(self.decoder(x))  # Output in [0, 1] range

class VAE(nn.Module):
    def __init__(self, seq_length, latent_channel, latent_dim):
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(seq_length, latent_channel, latent_dim)
        self.decoder = CNNDecoder(seq_length, latent_channel, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

# Custom VAE loss function
def vae_loss_function(reconstruction, original, mean, logvar, alpha):
    recon_loss = F.mse_loss(reconstruction, original)
    kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + alpha * kl_divergence, recon_loss, kl_divergence

# Training function with Early Stopping and Best Model Saving
def train(model, data_tensor, optimizer, num_epochs=300, alpha=0.1, patience=10, log_interval=10):
    logger.info("Initializing model...")
    logger.info("Initializing model layers...")
    
    model.train()
    total_loss_list, recon_loss_list, kl_divergence_list = [], [], []
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    logger.info("Starting training loop with early stopping...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstruction, mean, logvar = model(data_tensor.to(device))
        total_loss, recon_loss, kl_divergence = vae_loss_function(reconstruction, data_tensor.to(device), mean, logvar, alpha)
        
        total_loss.backward()
        optimizer.step()

        # Log detailed loss information
        total_loss_list.append(total_loss.item())
        recon_loss_list.append(recon_loss.item())
        kl_divergence_list.append(kl_divergence.item())

        # Check for early stopping and log best loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - New Best Loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        # Log current losses every log_interval epochs
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {total_loss.item():.4f}, "
                        f"Reconstruction Loss: {recon_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}")

    # Plot and save training curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_list, label='Total Loss')
    plt.plot(recon_loss_list, label='Reconstruction Loss')
    plt.plot(kl_divergence_list, label='KL Divergence')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"training_curves_{timestamp}.png")
    plt.show()

    # Save the best model at the end
    if best_model_state is not None:
        model_save_path = f"vae_model_best_{timestamp}.pth"
        torch.save(best_model_state, model_save_path)
        logger.info(f"Best model saved to {model_save_path} with loss {best_loss:.4f}")

    return model