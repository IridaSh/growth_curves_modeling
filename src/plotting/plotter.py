import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """
    Handles plotting of simulation results.
    """
    def __init__(self):
        """
        Initializes the Plotter.
        """
        pass

    def plot_individual_simulation(self, t, results, conditions, variables=['n', 'b', 'a'], num_sets_to_plot=10):
        """
        Plots individual simulation sets.

        Parameters:
        - t (ndarray): Array of time points.
        - results (ndarray): Simulation results with shape (num_simulations, len(conditions), 3, len(t)).
        - conditions (list of tuples): List of (a0, inh0) conditions.
        - variables (list of str): List of variable names to plot.
        - num_sets_to_plot (int): Number of simulation sets to plot.
        """
        num_simulations = results.shape[0]
        num_sets_to_plot = min(num_sets_to_plot, num_simulations)
        
        for i in range(num_sets_to_plot):
            fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            plt.subplots_adjust(hspace=0.4)
            
            for j, variable in enumerate(variables):
                ax = axs[j]
                for k, (a0, inh0) in enumerate(conditions):
                    label = f'Condition: a0={a0}, inh0={inh0}'
                    ax.semilogy(t, results[i, k, j], label=label, lw=1)
                
                ax.set_title(f'Simulation Set {i+1}: {variable}')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{variable} (log scale)')
                ax.set_ylim(bottom=1e-5)
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.show()

    def plot_aggregated_results(self, t, results, conditions, variables=['n', 'b', 'a']):
        """
        Plots aggregated simulation results (e.g., mean and confidence intervals).

        Parameters:
        - t (ndarray): Array of time points.
        - results (ndarray): Simulation results with shape (num_simulations, len(conditions), 3, len(t)).
        - conditions (list of tuples): List of (a0, inh0) conditions.
        - variables (list of str): List of variable names to plot.
        """
        num_conditions = len(conditions)
        num_variables = len(variables)
        
        fig, axs = plt.subplots(num_variables, 1, figsize=(12, 4*num_variables), sharex=True)
        plt.subplots_adjust(hspace=0.4)
        
        for j, variable in enumerate(variables):
            ax = axs[j]
            for k, (a0, inh0) in enumerate(conditions):
                # Extract data for the current condition and variable
                data = results[:, k, j, :]  # Shape: (num_simulations, len(t))
                
                # Compute mean and 95% confidence intervals
                mean = np.nanmean(data, axis=0)
                lower = np.nanpercentile(data, 2.5, axis=0)
                upper = np.nanpercentile(data, 97.5, axis=0)
                
                label = f'Condition: a0={a0}, inh0={inh0}'
                ax.semilogy(t, mean, label=label, lw=2)
                ax.fill_between(t, lower, upper, alpha=0.3)
            
            ax.set_title(f'Aggregated Results: {variable}')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{variable} (log scale)')
            ax.set_ylim(bottom=1e-5)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_time_series_comparison(self, t, curves, conditions, variables=['n', 'b', 'a'], num_sets_to_plot=10):
        """
        Plots time series comparisons across different conditions for multiple simulation sets.

        Parameters:
        - t (ndarray): Array of time points.
        - curves (ndarray): Simulation results with shape (num_simulations, len(conditions), 3, len(t)).
        - conditions (list of tuples): List of (a0, inh0) conditions.
        - variables (list of str): List of variable names to plot.
        - num_sets_to_plot (int): Number of simulation sets to plot.
        """
        num_simulations = curves.shape[0]
        num_sets_to_plot = min(num_sets_to_plot, num_simulations)
        
        for i in range(num_sets_to_plot):
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            plt.subplots_adjust(hspace=0.5)
            
            for j, variable in enumerate(variables):
                ax = axs[j]
                for k, (a0, inh0) in enumerate(conditions):
                    label = f'Condition: a0={a0}, inh0={inh0}'
                    ax.semilogy(t, curves[i, k, j], label=label, lw=1.5)
                
                ax.set_title(f'Simulation Set {i+1}: {variable}')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{variable} (log scale)')
                ax.set_ylim(bottom=1e-5)
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.show()