import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """Load data from an npz file."""
    return np.load(file_path)['curves']

def concatenate_time_series(data_array):
    """
    Concatenate the 'n' values for each sample across conditions.
    
    Parameters:
    - data_array: 4D numpy array containing time-series data.

    Returns:
    - all_n_values: 2D numpy array where each row represents a concatenated time series for a strain.
    """
    num_samples = data_array.shape[0]
    num_conditions = data_array.shape[1]
    num_time_points = data_array.shape[3]
    
    # Initialize an empty array to store all concatenated 'n' values
    all_n_values = np.empty((num_samples, num_conditions * num_time_points))

    # Concatenate time series for each strain across conditions
    for strain_index in range(num_samples):
        all_n_values[strain_index, :] = np.concatenate(
            [data_array[strain_index, condition_index, 0, :] for condition_index in range(num_conditions)]
        )
    
    return all_n_values

def plot_and_save_time_series(all_n_values, num_strains_to_display=10, save_dir='plots'):
    """
    Plot and save the concatenated time series for the first few strains.

    Parameters:
    - all_n_values: 2D numpy array where each row is a concatenated time series for a strain.
    - num_strains_to_display: int, number of strains to display and save plots for.
    - save_dir: str, directory to save the plots.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for strain_index in range(num_strains_to_display):
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Plot the concatenated 'n' curve
        ax.plot(all_n_values[strain_index, :])
        ax.set_xlabel('Index')
        ax.set_ylabel('n')
        ax.set_title(f'Strain {strain_index + 1}')
        
        # Save the figure
        file_path = os.path.join(save_dir, f'strain_{strain_index + 1}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
        
        print(f'Saved plot for Strain {strain_index + 1} at {file_path}')

def main():
    # Load data
    file_path = 'data/simulated/curves10k.npz'
    data_array = load_data(file_path)
    
    # Concatenate time series
    all_n_values = concatenate_time_series(data_array)
    
    # Plot and save time series
    plot_and_save_time_series(all_n_values)

    # Print the shape of the concatenated data for reference
    print("Shape of all_n_values:", all_n_values.shape)

# Run the main function
if __name__ == "__main__":
    main()