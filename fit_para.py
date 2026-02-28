import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from evolve import EV
from io_put import load_input

def load_psd(file_path):
    """
    Load PSD data from a file. 
    Returns (frequency, linear_psd).
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip potential header lines that aren't numeric
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    f_val = float(parts[0])
                    p_val = float(parts[1])
                    data.append([f_val, p_val])
                except ValueError:
                    continue
    if not data:
        raise ValueError(f"No valid data found in {file_path}")
    
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    
    freq = data[:, 0]
    psd = data[:, 1]
    
    # Robust check: log10 results are usually < 50 for realistic nm^4 surfaces.
    # Linear results (nm^4) are usually much larger than their log10 counterparts.
    if np.max(psd) > 40: # If any value > 40, it's likely linear (10^40 is physically impossible as log10)
        psd_linear = psd
    else:
        # Check if the values are all positive. If not, it's definitely log.
        if np.any(psd <= 0):
            psd_linear = 10**psd
        else:
            # If all positive but small, it's likely log.
            psd_linear = 10**psd
        
    return freq, psd_linear

def fit_parameters():
    print("=== Starting Parameter Fitting (c1, c2) ===")
    os.makedirs('result', exist_ok=True)
    report_file = open('result/fitting.out', 'w', encoding='utf-8')
    
    def log_print(msg):
        print(msg)
        report_file.write(msg + '\n')

    log_print(f"=== Parameter Fitting Report ===")
    
    # 1. Load configuration and experimental files
    try:
        config = load_input('input_fit.toml')
    except Exception as e:
        log_print(f"Error loading input_fit.toml: {e}")
        report_file.close()
        return
    
    # Path setup from TOML
    initial_file = config.get('psd_init')
    target_file = config.get('psd_final')
    
    if not initial_file or not target_file:
        log_print(f"Error: psd_init or psd_final not specified in input_fit.toml")
        report_file.close()
        return
    
    if not os.path.exists(initial_file) or not os.path.exists(target_file):
        log_print(f"Error: Experimental files not found. Check paths: {initial_file}, {target_file}")
        report_file.close()
        return

    # Load PSDs
    try:
        freq_exp_init, psd_exp_init_linear = load_psd(initial_file)
        freq_exp_target, psd_exp_target_linear = load_psd(target_file)
    except Exception as e:
        log_print(f"Error loading PSD files: {e}")
        report_file.close()
        return
    
    log_print(f"Loaded initial PSD: {initial_file} ({len(freq_exp_init)} points)")
    log_print(f"Loaded target PSD: {target_file} ({len(freq_exp_target)} points)")

    # 2. Setup Evolution
    ev = EV()
    T_initial = config.get('T_initial', 1600)
    T_anneal = config.get('T_anneal', 1608.15)
    anneal_time = config.get('anneal_time', 90)
    heating_rate = config.get('heating_rate', 50)
    cooling_rate = config.get('cooling_rate', 50)
    dt_max = config.get('dt_max', 1.0)

    # Build temperature profile
    from utils import compute_temperature_profile
    t_all, T_all = compute_temperature_profile(T_initial, T_anneal, anneal_time, heating_rate, cooling_rate)

    # Intersection of frequency ranges to avoid extrapolation artifacts
    f_min = max(freq_exp_init.min(), freq_exp_target.min())
    f_max = min(freq_exp_init.max(), freq_exp_target.max())
    
    # Interpolate target PSD onto initial frequencies for comparison
    interp_target = interp1d(freq_exp_target, np.log10(np.maximum(psd_exp_target_linear, 1e-30)), 
                             kind='linear', bounds_error=False, fill_value=np.nan)
    target_log_interpolated = interp_target(freq_exp_init)
    
    # Mask to only compare points within the common range
    fit_mask = (freq_exp_init >= f_min) & (freq_exp_init <= f_max) & (~np.isnan(target_log_interpolated))

    if not np.any(fit_mask):
        log_print("Error: No overlapping frequency range between initial and target data.")
        report_file.close()
        return

    # 3. Define Objective Function
    def objective(params):
        log_c1, log_c2 = params
        c1 = 10**log_c1
        c2 = 10**log_c2
        
        # Evolve with parameterized noise terms
        psd_sim_linear = ev.evolve_psd_1d(psd_exp_init_linear, freq_exp_init, T_all, t_all, dt_max, c1, c2)
        
        # Error calculation
        sim_log = np.log10(np.maximum(psd_sim_linear, 1e-30))
        mse = np.mean((sim_log[fit_mask] - target_log_interpolated[fit_mask])**2)
        
        log_print(f"  Iteration: log_c1={log_c1:6.2f}, log_c2={log_c2:6.2f} -> RMSE={np.sqrt(mse):.4f}")
        return mse

    # 4. Optimization
    initial_log_c1 = np.log10(max(config.get('c1', 1e-7), 1e-20))
    initial_log_c2 = np.log10(max(config.get('c2', 1e-4), 1e-20))
    
    x0 = [initial_log_c1, initial_log_c2]
    
    log_print(f"Starting Nelder-Mead optimization from x0: log_c1={initial_log_c1:.2f}, log_c2={initial_log_c2:.2f}")
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-2, options={'maxiter': 50})
    
    best_log_c1, best_log_c2 = res.x
    best_c1, best_c2 = 10**best_log_c1, 10**best_log_c2
    
    log_print("\n=== Fitting Results Summary ===")
    log_print(f"Best log10(c1): {best_log_c1:.4f} -> c1: {best_c1:.4e}")
    log_print(f"Best log10(c2): {best_log_c2:.4f} -> c2: {best_c2:.4e}")
    log_print(f"Status: {res.message}")

    # 5. Final Output and Plotting
    psd_best_linear = ev.evolve_psd_1d(psd_exp_init_linear, freq_exp_init, T_all, t_all, dt_max, best_c1, best_c2)
    
    plt.figure(figsize=(10, 7))
    plt.loglog(freq_exp_init, psd_exp_init_linear, 'k--', label='Initial (Exp)', alpha=0.6)
    plt.loglog(freq_exp_target, psd_exp_target_linear, 'ro', label='Experimental (Target)', markersize=4, alpha=0.6)
    plt.loglog(freq_exp_init, psd_best_linear, 'b-', linewidth=2, label=f'Best Fit (c1={best_c1:.1e}, c2={best_c2:.1e})')
    plt.xlabel('Frequency (/um)')
    plt.ylabel('PSD (nm^4)')
    plt.title('Parameter Fitting: Experimental vs Best-Fit Simulation')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)
    
    output_img = 'result/fitting_comparison.png'
    plt.savefig(output_img, dpi=150)
    log_print(f"Result plot saved to '{output_img}'")
    
    log_print("\nUse these parameters in input.toml:")
    log_print(f"c1 = {best_c1:.6e}")
    log_print(f"c2 = {best_c2:.6e}")
    
    report_file.close()

if __name__ == "__main__":
    fit_parameters()