"""Growth model implementation and simulation classes"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

class GrowthModel:
    """Growth model implementation"""
    def __init__(self, p: np.ndarray, inh: float):
        self.p = p
        self.inh = inh
        self.initiate_lysis = False
        
    def __call__(self, t: float, y: np.ndarray) -> List[float]:
        y = np.maximum(y, 0)
        n, b, a = y
        
        # Unpack model parameters
        mumax, Ks, theta, Ln, kappab, phimax, gamma, betamin, db, c = self.p
        
        # Constants
        db0, da0 = 0.001, 0.001
        ha, hi = 3, 3
        Ka, Ki = 1, 15
        Nm = 3.0
        
        # Calculate rates
        iota = (self.inh**hi) / (1 + self.inh**hi) if self.inh > 0 else 0
        beta = betamin + c * (1 - betamin) * iota
        phi = phimax * (1 - c * iota)
        
        try:
            g = (1 / (1 + (n / (Nm * Ks))**theta)) * (1 - (n / Nm)) if Ks > 0 else 0
        except RuntimeWarning:
            g = 0
            
        # Lysis calculation
        l = 0
        if (a > 0 or self.inh > 0):
            if not self.initiate_lysis and n > Ln:
                self.initiate_lysis = True
            if self.initiate_lysis:
                inh_term = (self.inh / Ki)**hi
                denominator = 1 + a**ha + inh_term
                if denominator > 0:
                    l = gamma * g * (a**ha + inh_term) / denominator
        
        growth_rate = mumax * g * n
        lysis_rate = beta * l * n
        
        denominator = Ka + a
        antibiotic_term = -(kappab * b + phi * n) * a / denominator if denominator > 0 else 0
        
        return [
            growth_rate - lysis_rate,
            lysis_rate - (db * iota + db0) * b,
            antibiotic_term - da0 * a
        ]

class Simulation:
    """Single simulation runner"""
    def __init__(self, p: np.ndarray, n0: float, a0: float, inh: float):
        self.p = p
        self.n0 = n0
        self.a0 = a0
        self.inh = inh
        self.b0 = 0
        self.y0 = np.array([n0, self.b0, a0])

    def run(self, t_span: Tuple[float, float] = (0, 24), 
            num_time_points: int = 145) -> Tuple[np.ndarray, np.ndarray]:
        t_eval = np.linspace(t_span[0], t_span[1], num_time_points)
        model = GrowthModel(self.p, self.inh)
        sol = solve_ivp(model, t_span, self.y0, t_eval=t_eval)
        return sol.t, sol.y

class ParameterSampler:
    """Parameter sampling functionality"""
    def __init__(self, param_specs: Dict, seed: Optional[int] = None):
        self.param_specs = param_specs
        if seed is not None:
            np.random.seed(seed)
    
    def sample_parameters(self) -> List[float]:
        p = []
        for key in ['alpha', 'Ks', 'Nm', 'theta', 'kappab', 'phimax', 'gamma', 'betamin', 'db', 'c']:
            spec = self.param_specs[key]
            value = self._sample_truncated_normal(
                spec['mean'], spec['sd'], spec['low'], spec['high']
            )
            p.append(value)
        return p
    
    @staticmethod
    def _sample_truncated_normal(mean: float, sd: float, 
                               low: float, high: float, 
                               size: Optional[int] = None) -> float:
        a, b = (low - mean) / sd, (high - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

class SimulationRunner:
    """Multiple simulation runner"""
    def __init__(self, num_simulations: int, n0: float, conditions: List[Tuple],
                 t_eval: np.ndarray, param_specs: Dict, 
                 logger: Optional[logging.Logger] = None,
                 seed: Optional[int] = None):
        self.num_simulations = num_simulations
        self.n0 = n0
        self.conditions = conditions
        self.t_eval = t_eval
        self.param_sampler = ParameterSampler(param_specs, seed=seed)
        self.logger = logger or logging.getLogger(__name__)
        self.parameters = np.zeros((num_simulations, 10))
        self.curves = np.zeros((num_simulations, len(conditions), 3, len(t_eval)))

    def simulate(self, p: np.ndarray, n0: float, 
                a0: float, inh0: float) -> np.ndarray:
        sim = Simulation(p, n0, a0, inh0)
        _, results = sim.run(t_span=(0, 24), num_time_points=len(self.t_eval))
        return results

    def run_all_simulations(self, save_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.info(f"Starting {self.num_simulations} simulations...")
        
        try:
            for i in range(self.num_simulations):
                # Sample parameters
                p = self.param_sampler.sample_parameters()
                self.parameters[i] = p
                
                # Run simulations for each condition
                for j, (a0, inh0) in enumerate(self.conditions):
                    self.curves[i, j] = self.simulate(p, self.n0, a0, inh0)
                
                if (i + 1) % 1000 == 0:
                    self.logger.info(f'Completed {i + 1}/{self.num_simulations} simulations')
            
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_dir / 'curves10k.npz', 
                                 curves=self.curves, t_eval=self.t_eval)
                np.save(save_dir / 'parameters.npy', self.parameters)
                self.logger.info(f"Results saved to {save_dir}")
            
            return self.curves, self.parameters
            
        except Exception as e:
            self.logger.error(f"Error in simulations: {str(e)}", exc_info=True)
            raise