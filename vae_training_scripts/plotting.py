# plotting.py
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

import torch

def plot_loss(train_losses, test_losses, output_dir):
    plt.figure(figsize=(6, 3))
    plt.semilogy(train_losses, label='Training')
    plt.semilogy(test_losses, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()

def plot_reconstructions(model, subset_train_data, subset_test_data, output_dir, device):
    with torch.no_grad():
        output_train, _, _ = model(subset_train_data)
        output_test, _, _ = model(subset_test_data)

    output_train = output_train.squeeze(1).cpu().numpy()
    output_test = output_test.squeeze(1).cpu().numpy()
    subset_train_data = subset_train_data.cpu().numpy().squeeze(1)
    subset_test_data = subset_test_data.cpu().numpy().squeeze(1)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Training data plot
    axs[0].scatter(subset_train_data.flatten(), output_train.flatten(),
                   s=0.1, color='blue', alpha=0.5)
    axs[0].plot([subset_train_data.min(), subset_train_data.max()],
                [subset_train_data.min(), subset_train_data.max()], 'r')
    axs[0].set_xlim(subset_train_data.min(), subset_train_data.max())
    axs[0].set_ylim(subset_train_data.min(), subset_train_data.max())
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlabel('Original')
    axs[0].set_ylabel('Reconstructed')
    axs[0].set_title('Training')
    r2_train = r2_score(subset_train_data.flatten(), output_train.flatten())
    axs[0].text(0.05, 0.95, f'R² = {r2_train:.3f}',
                transform=axs[0].transAxes, verticalalignment='top')

    # Testing data plot
    axs[1].scatter(subset_test_data.flatten(), output_test.flatten(),
                   s=0.1, color='blue', alpha=0.5)
    axs[1].plot([subset_test_data.min(), subset_test_data.max()],
                [subset_test_data.min(), subset_test_data.max()], 'r')
    axs[1].set_xlim(subset_test_data.min(), subset_test_data.max())
    axs[1].set_ylim(subset_test_data.min(), subset_test_data.max())
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlabel('Original')
    axs[1].set_ylabel('Reconstructed')
    axs[1].set_title('Testing')
    r2_test = r2_score(subset_test_data.flatten(), output_test.flatten())
    axs[1].text(0.05, 0.95, f'R² = {r2_test:.3f}',
                transform=axs[1].transAxes, verticalalignment='top')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'reconstruction_scatter.png')
    plt.savefig(scatter_path)
    plt.close()

def plot_sample_trajectories(model, subset_train_data, subset_test_data, output_dir, device):
    with torch.no_grad():
        output_train, _, _ = model(subset_train_data)
        output_test, _, _ = model(subset_test_data)

    output_train = output_train.squeeze(1).cpu().numpy()
    output_test = output_test.squeeze(1).cpu().numpy()
    subset_train_data = subset_train_data.cpu().numpy().squeeze(1)
    subset_test_data = subset_test_data.cpu().numpy().squeeze(1)

    fig, axs = plt.subplots(2, 5, figsize=(10, 6))

    # Training data trajectories
    for i in range(5):
        axs[0, i].plot(subset_train_data[i], label='Original', color='blue')
        axs[0, i].plot(output_train[i], label='Reconstructed', color='orange')
        axs[0, i].set_title(f'Training {i + 1}')
        axs[0, i].legend()

    # Testing data trajectories
    for i in range(5):
        axs[1, i].plot(subset_test_data[i], label='Original', color='blue')
        axs[1, i].plot(output_test[i], label='Reconstructed', color='orange')
        axs[1, i].set_title(f'Testing {i + 1}')
        axs[1, i].legend()

    plt.tight_layout()
    trajectories_path = os.path.join(output_dir, 'sample_trajectories.png')
    plt.savefig(trajectories_path)
    plt.close()