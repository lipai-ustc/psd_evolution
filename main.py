import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from io_put import (load_input, 
                   generate_and_calibrate_psd,
                   save_psd_to_file)

from utils import  (generate_surface_from_psd_1d,
                   compute_temperature_profile,
                   calculate_rms_from_h,
                   visualize_evolution_results)

from evolve import EV

print("\n=== Silicon Wafer Surface Thermal Evolution Simulator ===")
os.makedirs('result', exist_ok=True)

config = load_input()
target_rms =    config.get('target_rms')         # 目标初始RMS (nm)
scan_size_um =  config.get('scan_size_um')       # 扫描尺寸 (μm) - 使用全局参数
size =          config.get('size')               # 图像像素数

# 初始温度 (K) - 使用统一配置参数
T_initial=      config.get('T_initial')
T_anneal =      config.get('T_anneal')           # 退火温度 (K)
heating_rate =  config.get('heating_rate')       # 升温速率 (K/s)
cooling_rate =  config.get('cooling_rate')       # 降温速率 (K/s)
anneal_time =   config.get('anneal_time')        # 保温时间 (s)

dt_max =        config.get('dt_max')             # 演化时间步长上限 (s)
c1 =            config.get('c1', 1e-7)           # 噪声参数c1
c2 =            config.get('c2', 1e-4)           # 噪声参数c2

print(f"Simulation Parameters:")
print(f"  初始RMS: {target_rms} nm")
print(f"  退火温度: {T_anneal} K")
print(f"  保温时间: {anneal_time} s")
print(f"  噪声系数: c1={c1:.2e}, c2={c2:.2e}")

# Generate, synthesize, and calibrate initial PSD/Surface to match target RMS
# freq_1d_um and psd_1d_initial (in log10 scale)
freq_1d_um, psd_1d_initial, phase, height_initial = generate_and_calibrate_psd(
    target_rms, scan_size_um, size)

rms_initial = target_rms
t_all, T_all = compute_temperature_profile(
    T_initial, T_anneal, anneal_time, heating_rate, cooling_rate)

# Evolution must be performed on linear PSD values
print("Performing thermal evolution (linear scale)...")
ev = EV()  
# Convert log10 PSD to linear PSD for physical evolution
psd_1d_initial_linear = 10**psd_1d_initial
psd_1d_evolved_linear = ev.evolve_psd_1d(
    psd_1d_initial_linear, freq_1d_um, T_all, t_all, dt_max, c1, c2)

# Convert evolved PSD back to log10 for saving and synthesis
psd_1d_evolved = np.log10(np.maximum(psd_1d_evolved_linear, 1e-300))
psd_1d_evolved_df = pd.DataFrame({'freq': freq_1d_um, 'amp': psd_1d_evolved})    

print("Generating evolved surface from evolved 1D PSD...")
height_evolved = generate_surface_from_psd_1d(psd_1d_evolved_df,
    scan_size_um=scan_size_um, size=size, phase=phase)

# save data to file
print("Saving results to files...")
initial_psd_file = f"result/initial_psd_uniform_rms_{target_rms:.1f}nm.txt"
evolved_psd_file = f"result/evolved_psd1d_uniform_rms_{target_rms:.1f}nm_anneal_{T_anneal:.0f}K.txt"

save_psd_to_file(freq_1d_um, psd_1d_initial, initial_psd_file)
save_psd_to_file(freq_1d_um, psd_1d_evolved, evolved_psd_file)
np.savetxt("result/height_initial.txt", height_initial, fmt="%.6e", delimiter="\t")
np.savetxt("result/height_evolved.txt", height_evolved, fmt="%.6e", delimiter="\t")

# 计算演化后表面的RMS
rms_evolved = calculate_rms_from_h(height_evolved)
rms_reduction = rms_initial - rms_evolved
reduction_percent = (rms_reduction / rms_initial) * 100 if rms_initial > 0 else 0

print(f"  演化后表面RMS: {rms_evolved:.6f} nm")
print(f"  粗糙度降低: {rms_reduction:.6f} nm ({reduction_percent:.1f}%)")

print("Generating visualization results...")
visualize_evolution_results(t_all, T_all, freq_1d_um, psd_1d_initial, psd_1d_evolved, 
                            height_initial, height_evolved, config, output_path='result.png')

