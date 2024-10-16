import sys
import os

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.simulations.simulation_runner import SimulationRunner
from src.plotting.plotter import Plotter

def main():
    """
    Main entry point for running growth model simulations and plotting results.
    """
    # Define parameter specifications directly within the code
    param_specs = {
        'mumax':   {'mean': 1.2,  'sd': 4,   'low': 0.2, 'high': 3},
        'Ks':      {'mean': 0.2,  'sd': 0.5, 'low': 0.1, 'high': 0.6},
        'theta':   {'mean': 3,    'sd': 5,   'low': 0.5, 'high': 6},
        'Ln':      {'mean': 0.2,  'sd': 0.4, 'low': 0,   'high': 0.8},
        'kappab':  {'mean': 2,    'sd': 10,  'low': 0,   'high': 4},
        'phimax':  {'mean': 4,    'sd': 10,  'low': 0,   'high': 8},
        'gamma':   {'mean': 2,    'sd': 3.2, 'low': 0,   'high': 4},
        'betamin': {'mean': 0.5,  'sd': 10,  'low': 0,   'high': 1},
        'db':      {'mean': 2.5,  'sd': 3,   'low': 0.5, 'high': 5},
        'c':       {'mean': 0.15, 'sd': 0.05,'low': 0,   'high': 0.3}
    }
    
    # Simulation parameters
    num_simulations = 10000                  # Total number of simulations to run
    n0 = 0.03                                # Initial population
    conditions = [(0, 0), (10, 0), (10, 10)] # List of (a0, inh0) initial conditions
    t_eval = np.linspace(0, 24, 145)         # Time points from 0 to 24 hours
    seed = 42                                # Seed for reproducibility
    
    # Initialize the simulation runner with all configurations
    simulation_runner = SimulationRunner(
        num_simulations=num_simulations,      # Number of simulations
        n0=n0,                                # Initial population
        conditions=conditions,                # Initial conditions for a0 and inh0
        t_eval=t_eval,                        # Time points for simulation
        param_specs=param_specs,              # Parameter specifications
        seed=seed                             # Seed for random number generator
    )
    
    # Run all simulations
    simulation_runner.run_all_simulations()
    
    # Initialize the plotter
    plotter = Plotter()
    
    # Plot the first 10 simulation sets for individual analysis
    plotter.plot_individual_simulation(
        t=t_eval,
        results=simulation_runner.curves,
        conditions=conditions,
        num_sets_to_plot=10  # Number of simulation sets to plot
    )
    
    # Plot aggregated results (e.g., mean and confidence intervals)
    plotter.plot_aggregated_results(
        t=t_eval,
        results=simulation_runner.curves,
        conditions=conditions
    )

if __name__ == "__main__":
    main()