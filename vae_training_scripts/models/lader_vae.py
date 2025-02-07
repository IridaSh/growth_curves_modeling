# models/ladder_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LadderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels*2, 3, padding=1)
        )
        self.skip = nn.Conv1d(in_channels, out_channels*2, 1)

    def forward(self, x):
        return self.conv(x) + self.skip(x)

class LadderVAE(nn.Module):
    def __init__(self, latent_dims=[32, 16, 8], seq_length=435):
        super().__init__()
        self.encoder = nn.ModuleList([
            LadderBlock(1, 32),
            LadderBlock(64, 16),
            LadderBlock(32, 8)
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(8, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(16, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 1, 3, padding=1)
            )
        ])
        
        self.fc_mu = nn.ModuleList([
            nn.Linear(64*seq_length, latent_dims[0]),
            nn.Linear(32*seq_length, latent_dims[1]),
            nn.Linear(16*seq_length, latent_dims[2])
        ])
        
        self.fc_logvar = nn.ModuleList([
            nn.Linear(64*seq_length, latent_dims[0]),
            nn.Linear(32*seq_length, latent_dims[1]),
            nn.Linear(16*seq_length, latent_dims[2])
        ])

    def encode(self, x):
        latents = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            mu = self.fc_mu[i](x.flatten(1))
            logvar = self.fc_logvar[i](x.flatten(1))
            latents.append((mu, logvar))
            x = F.max_pool1d(x, 2)
        return latents[::-1]  # Reverse for decoder

    def decode(self, zs):
        x = None
        for i, (z, block) in enumerate(zip(zs, self.decoder)):
            if x is None:
                x = z
            else:
                x = x + z  # Residual connection
            x = block(x)
            if i < len(self.decoder)-1:
                x = F.interpolate(x, scale_factor=2)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latents = self.encode(x)
        zs = [self.reparameterize(mu, logvar) for mu, logvar in latents]
        reconstruction = self.decode(zs)
        return reconstruction, latents