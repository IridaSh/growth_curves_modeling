import logging
from models.vae_cnn import VAE, train
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load your simulation data
logger.info("Loading data...")
data = np.load('data/simulated/curves10k.npz')['curves']  # Adjust path if needed
data_tensor = torch.tensor(data, dtype=torch.float32)
data_tensor = data_tensor.view(-1, 3, 145)  # Adjust the shape to (num_samples, channels, time_series_length)
logger.info(f"Data loaded with shape: {data_tensor.shape}")

seq_length = 145
latent_channel = 128  # Increased latent channels for better capacity
latent_dim = 64       # Larger latent dimension
model = VAE(seq_length, latent_channel, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = train(model, data_tensor, optimizer, num_epochs=50, alpha=0.1, patience=25)