import numpy as np
from scipy.integrate import solve_ivp
from src.models.growth_model import GrowthModel

class Simulation:
    """
    Handles individual simulation runs of the growth model.
    """
    def __init__(self, p, n0, a0, inh):
        """
        Initializes the simulation with model parameters and initial conditions.

        Parameters:
        - p (list or array): Model parameters.
        - n0 (float): Initial population.
        - a0 (float): Initial antibiotic concentration.
        - inh (float): Initial inhibitor concentration.
        """
        self.p = p  # Model parameters
        self.n0 = n0  # Initial population
        self.a0 = a0  # Initial antibiotic concentration
        self.inh = inh  # Initial inhibitor concentration
        self.b0 = 0  # Initial Bla concentration (assumed zero)
        self.y0 = np.array([n0, self.b0, a0])  # Initial state

    def run(self, t_span=(0, 24), num_time_points=145):
        """
        Runs the simulation over the specified time span.

        Parameters:
        - t_span (tuple): Start and end times for the simulation.
        - num_time_points (int): Number of time points to evaluate.

        Returns:
        - t (ndarray): Array of time points.
        - y (ndarray): Simulation results for population, Bla, and antibiotic concentration.
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_time_points)
        model = GrowthModel(self.p, self.inh)
        sol = solve_ivp(model, t_span, self.y0, t_eval=t_eval, method='RK45', vectorized=False)
        
        if not sol.success:
            raise RuntimeError(f"Simulation failed: {sol.message}")
        
        return sol.t, sol.y