"""Growth model implementation and simulation classes"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

class GrowthModel:
    def __init__(self, p, inh):
        self.p = p  # Model parameters
        self.inh = inh  # Inhibitor concentration
        self.initiate_lysis = False  # Track lysis initiation

    def __call__(self, t, y):
        y = np.maximum(y, 0)  # Ensure non-negative values
        n, b, a = y  # population (n), Bla (b), antibiotic (a)
        
        # Unpack model parameters
        mumax, Ks, theta, Ln, kappab, phimax, gamma, betamin, db, c = self.p

        # Constants for basal degradation and hill coefficients
        db0 = 0.001
        da0 = 0.001
        ha = 3
        hi = 3
        Ka = 1
        Ki = 15
        Nm = 3.0

        inh = self.inh
        iota = (inh**hi) / (1 + inh**hi) if inh > 0 else 0
        beta = betamin + c * (1 - betamin) * iota
        phi = phimax * (1 - c * iota)

        # Growth rate function with inhibition logic
        g = (1 / (1 + (n / (Nm * Ks))**theta)) * (1 - (n / Nm)) if Ks > 0 else 0

        # Lysis initiation
        l = 0
        if a > 0 or inh > 0:
            if not self.initiate_lysis and n > Ln:
                self.initiate_lysis = True
            if self.initiate_lysis:
                l = gamma * g * (a**ha + (inh / Ki)**hi) / (1 + a**ha + (inh / Ki)**hi)

        # Growth and lysis rates
        growth_rate = mumax * g * n
        lysis_rate = beta * l * n

        # Differential equations
        dndt = growth_rate - lysis_rate
        dbdt = lysis_rate - (db * iota + db0) * b
        dadt = -(kappab * b + phi * n) * a / (Ka + a) - da0 * a
        return [dndt, dbdt, dadt]

# Class for handling the simulation
class Simulation:
    def __init__(self, p, n0, a0, inh):
        self.p = p  # Model parameters
        self.n0 = n0  # Initial population
        self.a0 = a0  # Initial antibiotic concentration
        self.inh = inh  # Initial inhibitor concentration
        self.b0 = 0  # Initial Bla concentration (assumed zero)
        self.y0 = np.array([n0, self.b0, a0])  # Initial state

    def run(self, t_span=(0, 24), num_time_points=145):
        t_eval = np.linspace(t_span[0], t_span[1], num_time_points)
        model = GrowthModel(self.p, self.inh)
        sol = solve_ivp(model, t_span, self.y0, t_eval=t_eval)
        return sol.t, sol.y
    
# Function to sample from a truncated normal distribution
def sample_truncated_normal(mean, sd, low, upp, size=None):
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

class ParameterSampler:
    def __init__(self, param_specs, seed=None):
        self.param_specs = param_specs
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

    def sample_parameters(self):
        p = []
        for key in ['mumax', 'Ks', 'theta', 'Ln', 'kappab', 'phimax', 'gamma', 'betamin', 'db', 'c']:
            spec = self.param_specs[key]
            value = sample_truncated_normal(spec['mean'], spec['sd'], spec['low'], spec['high'])
            p.append(value)
        return p

# Class to handle simulation runs
class SimulationRunner:
    def __init__(self, num_simulations, n0, conditions, t_eval, param_specs, seed=None):
        self.num_simulations = num_simulations
        self.n0 = n0
        self.conditions = conditions
        self.t_eval = t_eval
        self.param_sampler = ParameterSampler(param_specs, seed=seed)
        self.parameters = np.zeros((num_simulations, 10))
        self.curves = np.zeros((num_simulations, len(conditions), 3, len(t_eval)))

    def simulate(self, p, n0, a0, inh0):
        # Define your growth model and the simulation logic
        sim = Simulation(p, n0, a0, inh0)
        _, results = sim.run(t_span=(0, 24), num_time_points=len(self.t_eval))
        return results

    def run_all_simulations(self):
        for i in range(self.num_simulations):
            # Sample parameters for this simulation
            p = self.param_sampler.sample_parameters()
            self.parameters[i] = p

            # Run simulations under the specified conditions
            for j, (a0, inh0) in enumerate(self.conditions):
                simulated = self.simulate(p, self.n0, a0, inh0)
                self.curves[i, j] = simulated

            if (i + 1) % 1000 == 0:
                print(f'Simulations completed: {i + 1}/{self.num_simulations}')

        # Save results to files
        np.savez_compressed('data/simulated/curves10k.npz', curves=self.curves, t_eval=self.t_eval)
        np.save('data/simulated/parameters.npy', self.parameters)