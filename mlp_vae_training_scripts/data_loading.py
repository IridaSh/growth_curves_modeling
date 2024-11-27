# data_loading.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

def load_data(file_path):
    """Load data from an npz file."""
    try:
        data = np.load(file_path)['curves']
        logging.info(f'Data loaded from {file_path} with shape {data.shape}')
        return data
    except Exception as e:
        logging.error(f'Error loading data from {file_path}: {e}')
        raise

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

    all_n_values = np.empty((num_samples, num_conditions * num_time_points))

    for strain_index in range(num_samples):
        all_n_values[strain_index, :] = np.concatenate(
            [data_array[strain_index, condition_index, 0, :] for condition_index in range(num_conditions)]
        )

    logging.info(f'Concatenated time series with shape {all_n_values.shape}')
    return all_n_values

def round_parameters(parameters, decimals=3):
    """
    Rounds the given numpy array to a specified number of decimal places.

    Parameters:
    - parameters: numpy array, the array of parameters to be rounded.
    - decimals: int, the number of decimal places to round to (default is 3).

    Returns:
    - numpy array with values rounded to the specified decimal places.
    """
    rounded = np.round(parameters, decimals=decimals)
    logging.info(f'Parameters rounded to {decimals} decimals.')
    return rounded

def normalize_data(data_file, parameter_file):
    """Normalize the input data and parameters."""
    data_array = load_data(data_file)

    # Concatenate time series
    data = concatenate_time_series(data_array)
    scaler_data = MinMaxScaler()
    normalized_data = scaler_data.fit_transform(data)
    logging.info('Data normalized using MinMaxScaler.')

    # Load and round parameters
    parameters = np.load(parameter_file)
    logging.info(f'Parameters shape: {parameters.shape}')
    logging.info(f'Max parameter values before rounding: {np.max(parameters, axis=0)}')

    # Round the parameters
    parameters = round_parameters(parameters, decimals=3)
    logging.info(f'Max parameter values after rounding: {np.max(parameters, axis=0)}')
    parameter_names = ['alpha','Ks','Nm', 'theta', 'kappab', 'phimax', 'gamma', 'betamin', 'db', 'c']

    # Dynamically calculate parameter_scale as the rounded-up maximum values
    parameter_scale = np.ceil(np.max(parameters, axis=0) * 1000) / 1000  # Round up to the nearest 0.001
    logging.info(f'Parameter scale (rounded-up max values): {parameter_scale}')

    # Normalize parameters by dividing by parameter_scale
    normalized_parameters = parameters / parameter_scale
    logging.info('Parameters normalized by dividing with parameter_scale.')

    input_size = normalized_data.shape[1]
    output_size = normalized_parameters.shape[1]

    return normalized_data, normalized_parameters, input_size, output_size, scaler_data, parameter_scale, parameter_names