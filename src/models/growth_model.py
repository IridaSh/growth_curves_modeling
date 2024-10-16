import numpy as np


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