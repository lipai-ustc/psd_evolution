# PSD Simulation & Parameter Fitting Project

This project simulates the thermal evolution of silicon wafer surfaces using the linear Mullins-Herring (M-H) equation. It includes tools for generating Power Spectral Density (PSD) from experimental data, simulating surface morphology changes, and fitting physical parameters ($c_1, c_2$) to match experimental results.

## Project Structure

- `main.py`: The entry point for running a full simulation. Loads configuration, generates initial surfaces, evolves them thermally, and visualizes results.
- `evolve.py`: Core physical engine. Implements the 1D PSD evolution with temperature-dependent coefficients $v'(T)$ and parameterized noise.
- `fit_para.py`: Optimization script to fit noise coefficients $c_1$ and $c_2$ to experimental measurements.
- `utils.py`: Utility functions for surface synthesis, RMS calculation, temperature profiles, and result visualization.
- `io_put.py`: Handles file I/O, configuration loading, and GPR-based experimental PSD interpolation.
- `input.toml`: Configuration file for the main simulation.
- `input_fit.toml`: Configuration file for the parameter fitting process.

## Core Physical Model

The surface evolution follows the equation:
$$\frac{\partial h}{\partial t} = -v'(T) \nabla^4 h + \eta(\mathbf{r}, t)$$

The Power Spectral Density (PSD) evolution is implemented as:
$$S(f, t+dt) = S(f, t) e^{-2 v' f^4 dt} + \frac{\Delta + \gamma f^2}{(2\pi)^4 (2\pi f)} \frac{1 - e^{-2 v' f^4 dt}}{2 v' f^4}$$

Where:
- $\Delta = c_1 \cdot v'(T) \cdot T$ (Non-conservative noise / Etching)
- $\gamma = c_2 \cdot v'(T) \cdot T$ (Conservative noise / Thermal fluctuations)
- $v'(T)$ is the temperature-dependent diffusion coefficient.
- The factor $2\pi f$ ensures consistency between 1D and 2D isotropic PSD models.

## How to Use

### 1. Parameter Fitting
To find the best noise coefficients for your experimental data:
1. Update `input_fit.toml` with the paths to your initial (`psd_init`) and target (`psd_final`) experimental PSD files.
2. Run the script:
   ```bash
   python fit_para.py
   ```
3. Results will be saved to `result/fitting.out` and a comparison plot to `result/fitting_comparison.png`.

### 2. Running Simulation
To simulate a surface evolution with specific parameters:
1. Update `input.toml` with your desired RMS, temperature profile, and the optimized `c1`, `c2` values.
2. Run the simulation:
   ```bash
   python main.py
   ```
3. Comprehensive results including temperature profiles, PSD shifts, and surface topography will be saved to `result.png`.

## Requirements
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tomli`, `scikit-learn`
