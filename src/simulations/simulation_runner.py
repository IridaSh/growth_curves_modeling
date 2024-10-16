import numpy as np
import os
from src.simulations.parameter_sampler import ParameterSampler
from src.simulations.simulation import Simulation

class SimulationRunner:
    """
    Manages multiple simulation runs, handles parameter sampling, and saves results.
    """
    def __init__(self, num_simulations, n0, conditions, t_eval, param_specs, seed=None):
        """
        Initializes the simulation runner with the necessary configurations.

        Parameters:
        - num_simulations (int): Number of simulations to run.
        - n0 (float): Initial population.
        - conditions (list of tuples): List of (a0, inh0) conditions for simulations.
        - t_eval (ndarray): Array of time points to evaluate.
        - param_specs (dict): Specifications for parameter sampling.
        - seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.num_simulations = num_simulations
        self.n0 = n0
        self.conditions = conditions
        self.t_eval = t_eval
        self.param_sampler = ParameterSampler(param_specs, seed=seed)
        self.parameters = np.zeros((num_simulations, len(param_specs)))
        self.curves = np.zeros((num_simulations, len(conditions), 3, len(t_eval)))  # 3 variables: n, b, a

        # Ensure the simulated data directory exists
        self.simulated_data_dir = os.path.join('data', 'simulated')
        os.makedirs(self.simulated_data_dir, exist_ok=True)

    def simulate(self, p, a0, inh0):
        """
        Runs a single simulation with given parameters and conditions.

        Parameters:
        - p (list): Model parameters.
        - a0 (float): Initial antibiotic concentration.
        - inh0 (float): Initial inhibitor concentration.

        Returns:
        - results (ndarray): Simulation results for population, Bla, and antibiotic concentration.
        """
        sim = Simulation(p, self.n0, a0, inh0)
        _, results = sim.run(t_span=(0, 24), num_time_points=len(self.t_eval))
        return results

    def run_all_simulations(self):
        """
        Runs all simulations, samples parameters, and saves the results.
        """
        for i in range(self.num_simulations):
            # Sample parameters for this simulation
            p = self.param_sampler.sample_parameters()
            self.parameters[i] = p

            # Run simulations under the specified conditions
            for j, (a0, inh0) in enumerate(self.conditions):
                try:
                    simulated = self.simulate(p, a0, inh0)
                    self.curves[i, j] = simulated
                except RuntimeError as e:
                    print(f"Simulation {i+1}, condition {j+1} failed: {e}")
                    self.curves[i, j] = np.nan  # Assign NaN to indicate failure

            # Progress logging
            if (i + 1) % 1000 == 0 or (i + 1) == self.num_simulations:
                print(f'Simulations completed: {i + 1}/{self.num_simulations}')

        # Save results to files
        curves_path = os.path.join(self.simulated_data_dir, 'curves10k.npz')
        parameters_path = os.path.join(self.simulated_data_dir, 'parameters.npy')
        np.savez_compressed(curves_path, curves=self.curves, t_eval=self.t_eval)
        np.save(parameters_path, self.parameters)
        print(f"Simulation results saved to '{curves_path}' and '{parameters_path}'.")