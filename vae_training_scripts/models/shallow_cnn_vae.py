import torch.nn as nn
import torch.nn.functional as F
from .base_vae import BaseVAE

class CNNEncoder(nn.Module):
    """CNN Encoder for VAE Architecture 1."""
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
    """CNN Decoder for VAE Architecture 1."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
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

class VAE(BaseVAE):
    """VAE using CNN Architecture 1."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(VAE, self).__init__(latent_dim, latent_channel, seq_length)
        self.encoder = CNNEncoder(latent_dim, latent_channel, seq_length)
        self.decoder = CNNDecoder(latent_dim, latent_channel, seq_length)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar