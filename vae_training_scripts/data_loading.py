# data_loading.py
import os
import numpy as np

def load_simulated_data(data_directory, logger):
    """Load and process simulated data."""
    data_path = os.path.join(data_directory, 'curves10k.npz')
    logger.info(f"Loading simulated data from {data_path}")
    data_array = np.load(data_path)['curves']
    
    conditions = [(0, 0), (10, 0), (10, 10)]
    num_strains, num_conditions, _, seq_length = data_array.shape
    all_n_values = np.empty((num_strains, num_conditions * seq_length))
    
    for strain_index in range(num_strains):
        concatenated_values = np.concatenate(
            [data_array[strain_index, condition_index, 0, :] for condition_index in range(num_conditions)]
        )
        all_n_values[strain_index, :] = concatenated_values

    logger.info(f"Simulated data shape: {all_n_values.shape}")
    return all_n_values

def load_isolates_data(data_path, logger):
    """Load and process isolates data."""
    logger.info(f"Loading isolates data from {data_path}")
    data_array = np.load(data_path)['arr_0']
    
    conditions = [(0, 0), (50, 0), (50, 25)]
    num_strains, _, num_replicates, seq_length = data_array.shape
    num_conditions = len(conditions)
    
    all_od_values = np.empty((num_strains, num_replicates, num_conditions * seq_length))
    
    for strain_index in range(num_strains):
        for replicate_index in range(num_replicates):
            concatenated_values = np.concatenate(
                [data_array[strain_index, condition_index, replicate_index, :] for condition_index in range(num_conditions)]
            )
            all_od_values[strain_index, replicate_index, :] = concatenated_values

    logger.info(f"Isolates data shape: {all_od_values.shape}")
    reshaped_od_values = all_od_values.reshape(-1, all_od_values.shape[2])
    return reshaped_od_values