# Bacterial Growth Simulation

## Overview

The **Bacterial Growth Simulation** project models the dynamics of bacterial populations under varying antibiotic and inhibitor concentrations. It utilizes differential equations, parameter sampling, and visualization tools to analyze bacterial growth and resistance mechanisms.

The Growth Curves Modeling with VAE and MLP project aims to leverage Variational Autoencoders (VAE) combined with Multi-Layer Perceptrons (MLP) to predict biological parameters from growth curve data. This project is structured into two main components:
	1.	VAE Training Scripts (vae_training_scripts/): Handles the training of different VAE architectures.
	2.	MLP and VAE Combined Training Scripts (mlp_vae_training_scripts/): Utilizes the trained VAE to extract latent representations and trains an MLP to predict parameters based on these representations.

By maintaining a modular and organized structure, the project ensures scalability, maintainability, and ease of experimentation with different model architectures.

## File Structure
```
growth_curves_modeling/
│
├── data/                       # Contains input datasets (e.g., simulated and isolates)
│
├── evaluation_results/         # Results from model evaluation for VAE
│
├── logs/                       # Log files from model evaluation
│
├── mlp_vae_training_scripts/   # Scripts for training MLP and VAE models
├── train_vae_output/       	# Output results and checkpoints from VAE training
├── train_vae_mlp_output/   	# Combined VAE-MLP training output
│
├── nonlinear_methods/          # Nonlinear optimization methods for parameter estimation
│
├── vae_training_scripts/       # Scripts for VAE model training
│
├── simulate_data.py            # Script to simulate growth curve data
│
├── simulation.ipynb            # Jupyter Notebook for growth curve simulations
├── pyproject.toml              # Project dependencies and settings
├── uv.lock                     # Lock file for uv (dependency management)
│
└── README.md                   # Project documentation (you're reading this!)
```


## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Git**

### Clone the Repository

```bash
git clone https://github.com/IridaSh/growth_curves_modeling.git
```
# Create and activate the virtual environment
```bash
uv venv
```

# Sync dependecies
```bash
uv sync
```
Dependencies are listed in pyproject.toml.

## Usage

### Training the Model

The project allows training different VAE architectures combined with an MLP for parameter prediction. The main scripts are located in vae_training_scripts/ and mlp_vae_training_scripts/.

VAE Training

To train a specific VAE model (shallow_cnn_vae, deep_cnn_vae, or residual_cnn_vae), navigate to the vae_training_scripts/ directory and run main.py with the desired model choice.
```bash
python main.py ----model DeepCNNVAE
```
Arguments:
- **`--model`**: Choose the VAE architecture to train. Options include:
  - `shallow_cnn_vae`  
  - `deep_cnn_vae`  
  - `residual_cnn_vae`  

Output is saved in train_vae_output/.

2. MLP and VAE Combined Training

After training the desired VAE model, use the combined training scripts to train the MLP based on the VAE’s latent representations.
```bash
cd ../mlp_vae_training_scripts/
python main.py --model VAE
```
Arguments:
- **`--model`**: Choose the VAE architecture used. Must match the one trained earlier.

Output is saved in train_vae_mlp_output/.

## Evaluation
To run the MSE for the VAE model navigate to vae_training_scripts and 
```bash
python vae_training_scripts/evaluate_model.py \
    --model-type VAE \
    --model-path train_vae_output/VAE_LD10_LC16_LR0.001_TS20241126_002111//models/best_model.pt \
    --output-dir evaluation_results \
    --log-dir logs \
    --latent-dim 10 \
    --latent-channel 16 \
    --batch-size 32
```
