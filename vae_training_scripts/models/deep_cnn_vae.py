import torch.nn as nn
import torch.nn.functional as F
from .base_vae import BaseVAE

class DeepCNNEncoder(nn.Module):
    """Deeper CNN Encoder for VAE"""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(DeepCNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, latent_channel, kernel_size=3, stride=1, padding=1),
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

class DeepCNNDecoder(nn.Module):
    """Deeper CNN Decoder for VAE"""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(DeepCNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
        
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.latent_channel, self.seq_length)
        x = self.decoder(x)
        return x

class DeepCNNVAE(BaseVAE):
    """Deeper Variational Autoencoder using CNN architecture"""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(DeepCNNVAE, self).__init__(latent_dim, latent_channel, seq_length)
        self.encoder = DeepCNNEncoder(latent_dim, latent_channel, seq_length)
        self.decoder = DeepCNNDecoder(latent_dim, latent_channel, seq_length)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar