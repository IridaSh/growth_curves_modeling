# vae_training_scripts/models/transformer_vae.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer inputs:
    [batch_size, seq_len, d_model], with batch_first=True
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sin/cos to even/odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as buffer => not a parameter, but on the correct device
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model] (batch_first=True)
        We add positional encodings up to x.size(1).
        """
        seq_len = x.size(1)
        # self.pe[:, :seq_len, :] => shape [1, seq_len, d_model]
        # This broadcasts to [batch, seq_len, d_model]
        return x + self.pe[:, :seq_len, :]


class TransformerVAE(nn.Module):
    """
    Transformer-based VAE that uses batch_first=True.

    - Input shape: [batch, 1, seq_len]
    - Internal shape for the Transformer: [batch, seq_len, embed_dim]
    """

    def __init__(self, latent_dim=10, seq_length=435, nhead=4, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.embed_dim = 64

        # --- ENCODER ---
        # 1) Convolution to get an embedding from the single input channel
        #    Input is [batch, 1, seq_len] => Output is [batch, embed_dim, seq_len]
        self.input_conv = nn.Conv1d(
            in_channels=1, out_channels=self.embed_dim,
            kernel_size=3, padding=1
        )

        # We'll use a TransformerEncoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True  # <--- crucial
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Positional encoding in shape [batch, seq_len, embed_dim]
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=seq_length)

        # Projection to latent space
        self.fc_mean = nn.Linear(self.embed_dim * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(self.embed_dim * seq_length, latent_dim)

        # --- DECODER ---
        # Re-expand from latent space -> [batch, seq_len, embed_dim]
        self.fc_z_to_embed = nn.Linear(latent_dim, self.embed_dim * seq_length)

        # Another Transformer for decoding
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True  # <--- crucial
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Positional encoding for the decoder input
        self.pos_enc_dec = PositionalEncoding(self.embed_dim, max_len=seq_length)

        # Final conv to go back to 1 channel
        # The decoder will output [batch, seq_len, embed_dim]
        # => we transpose to [batch, embed_dim, seq_len] => 1D conv
        self.output_conv = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def encode(self, x):
        """
        x: [batch, 1, seq_len]
        Returns: mean, logvar => both [batch, latent_dim]
        """
        # 1) Convolution => [batch, embed_dim, seq_len]
        x = self.input_conv(x)

        # 2) Permute to [batch, seq_len, embed_dim] because we use batch_first=True
        x = x.permute(0, 2, 1)  # from [batch, embed_dim, seq_len]

        # 3) Add positional encoding => still [batch, seq_len, embed_dim]
        x = self.pos_enc(x)

        # 4) Transformer encoder => [batch, seq_len, embed_dim]
        x = self.transformer_encoder(x)

        # 5) Flatten => [batch, seq_len * embed_dim]
        x = x.reshape(x.size(0), -1)

        # 6) Mean & logvar
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        """
        z: [batch, latent_dim]
        Should produce a reconstruction => [batch, 1, seq_len]
        """
        # 1) Re-expand => [batch, seq_len * embed_dim]
        x = self.fc_z_to_embed(z)

        # 2) Reshape => [batch, seq_len, embed_dim]
        x = x.view(x.size(0), self.seq_length, self.embed_dim)

        # 3) Positional encoding
        x = self.pos_enc_dec(x)

        # 4) For a typical TransformerDecoder, we need a "memory" from the encoder.
        #    But here, we can pass the same x as "tgt" *and* "memory",
        #    or we can pass zero memory. We'll pass x as memory for demonstration.
        x = self.transformer_decoder(tgt=x, memory=x)

        # 5) We now have [batch, seq_len, embed_dim]
        #    Transpose to [batch, embed_dim, seq_len] for 1D convolution
        x = x.permute(0, 2, 1)
        # 6) Final conv => [batch, 1, seq_len]
        x = self.output_conv(x)
        return F.relu(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: [batch, 1, seq_len]
        Returns: reconstruction, mean, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def encoder(self, x):
        """
        Mimic the 'encoder(...)' call used by the CNN-based VAEs
        by simply calling self.encode(...).
        """
        return self.encode(x)