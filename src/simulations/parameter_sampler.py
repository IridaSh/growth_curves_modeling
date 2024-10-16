# src/simulations/parameter_sampler.py

import numpy as np
from src.utils.sampling import sample_truncated_normal

class ParameterSampler:
    """
    Samples model parameters from specified truncated normal distributions.
    """
    def __init__(self, param_specs, seed=None):
        """
        Initializes the parameter sampler with parameter specifications.

        Parameters:
        - param_specs (dict): Specifications for each parameter, including mean, sd, low, and high.
        - seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.param_specs = param_specs
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

    def sample_parameters(self):
        """
        Samples a set of parameters based on the provided specifications.

        Returns:
        - p (list): List of sampled parameter values in the order defined by param_specs.
        """
        p = []
        for key in ['mumax', 'Ks', 'theta', 'Ln', 'kappab', 'phimax', 'gamma', 'betamin', 'db', 'c']:
            spec = self.param_specs[key]
            value = sample_truncated_normal(spec['mean'], spec['sd'], spec['low'], spec['high'])
            p.append(value)
        return p