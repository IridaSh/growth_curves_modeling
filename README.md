# Bacterial Growth Simulation

## Overview

The **Bacterial Growth Simulation** project models the dynamics of bacterial populations under varying antibiotic and inhibitor concentrations. It utilizes differential equations, parameter sampling, and visualization tools to analyze bacterial growth and resistance mechanisms.

## Features

- **Growth Model Dynamics**: Simulate bacterial population growth with interactions involving antibiotics and inhibitors.
- **Parameter Sampling**: Explore a wide range of model parameters using truncated normal distributions.
- **Batch Simulations**: Run thousands of simulations efficiently to analyze different scenarios.
- **Visualization**: Generate comprehensive plots to visualize population dynamics and concentration changes.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Git**

### Clone the Repository

```bash
git clone https://github.com/IridaSh/growth_curves_modeling.git
```
# Create and activate the virtual environment
uv create .venv

# Install dependencies
uv install

# Running Simulations

Execute the main script to perform simulations and generate plots:
```bash
uv run src.main.py
```
