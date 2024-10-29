from models.autoencoders import SimpleVAE
import numpy as np
import torch
from torch.optim import Adam
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load your simulation data
logger.info("Loading data...")
data = np.load('data/simulated/curves10k.npz')['curves']  # Adjust path if needed
data_tensor = torch.tensor(data, dtype=torch.float32)
data_tensor = data_tensor.view(-1, 3, 145)  # Adjust the shape to (num_samples, channels, time_series_length)
logger.info(f"Data loaded with shape: {data_tensor.shape}")


# Define hyperparameters
hp = {
    "in": 3,
    "embedding_dimension": 32,
    "kernel_size": 3,
    "time_series_length": 145
}

# Initialize model, optimizer, and training parameters
logger.info("Initializing model...")
vae = SimpleVAE(hp)
optimizer = Adam(vae.parameters(), lr=0.001)
num_epochs = 300
alpha = 0.1
gamma = 0.0
early_stopping_patience = 20
best_loss = float('inf')
epochs_without_improvement = 0

# Tracking losses for plotting
total_loss_list = []
recon_loss_list = []
kl_divergence_list = []

# Training loop with early stopping
logger.info("Starting training loop with early stopping...")
for epoch in range(num_epochs):
    vae.train()
    optimizer.zero_grad()
    
    X_hat, code, mu, log_var = vae(data_tensor)
    total_loss, recon_loss, kl_divergence = SimpleVAE.vae_loss(X_hat, data_tensor, mu, log_var, alpha, gamma)
    
    total_loss.backward()
    optimizer.step()

    # Record losses
    total_loss_list.append(total_loss.item())
    recon_loss_list.append(recon_loss.item())
    kl_divergence_list.append(kl_divergence.item())

    # Early stopping check
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        epochs_without_improvement = 0
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - New Best Loss: {best_loss:.4f}")
    else:
        epochs_without_improvement += 1
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {total_loss.item():.4f} - No improvement for {epochs_without_improvement} epochs")

    if epochs_without_improvement >= early_stopping_patience:
        logger.info("Early stopping triggered. Stopping training.")
        break

    # Regular logging every 10 epochs
    if (epoch + 1) % 10 == 0:
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {total_loss.item():.4f}')

# Save model (optional)
logger.info("Saving the model...")
torch.save(vae.state_dict(), "vae_model.pth")
logger.info("Model saved successfully.")

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(total_loss_list, label='Total Loss')
plt.plot(recon_loss_list, label='Reconstruction Loss (BCE)')
plt.plot(kl_divergence_list, label='KL Divergence')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)
plt.show()