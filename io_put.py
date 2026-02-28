import os,re
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from utils import calculate_rms_from_psd_1d

def load_input(input_file='input.toml'):
    import tomli
    with open(input_file,'rb') as f:
        config = tomli.load(f)
    return config

def load_experimental_data_from_folder(data_dir="data"):
    """
    Load all experimental data files from a folder and extract PSD data.
    
    Parameters:
        data_dir: Path to the folder containing AFM PSD data files
    
    Returns:
        list: Experimental data including frequencies and PSD values
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create the directory and place AFM PSD files.")

    experimental_data = []
    seen_rms = set()  # 用于记录已加载的 RMS 值，防止重复
    print(f"正在扫描目录: {data_dir}")
    
    for filename in sorted(os.listdir(data_dir)):
        # 只处理.txt文件
        if not filename.lower().endswith('.txt'):
            continue
        filepath = os.path.join(data_dir, filename)
        
        try:
            # 尝试不同编码方式读取文件
            for encoding in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"X 读取 {filename} 时出错: 所有编码尝试都失败")
                continue
            if not lines:
                continue

            # 解析第一行的 RMS 值
            first_line = lines[0].strip()
            rms_match = re.search(r'RMS[:=：]\s*(\d+\.?\d*)', first_line, re.IGNORECASE)
            if rms_match:
                rms_val = float(rms_match.group(1))
            else:
                # 如果第一行是纯数字，直接解析为RMS
                try:
                    rms_val = float(first_line)
                except ValueError:
                    print(f"! 跳过 {filename}: 无法从第一行解析RMS值: '{first_line}'")
                    continue

            # === 关键修改：跳过重复 RMS ===
            if rms_val in seen_rms:
                print(f"! 跳过重复 RMS={rms_val:.2f} nm 的文件: {filename}")
                continue
            seen_rms.add(rms_val)  # 记录已加载的RMS

            # 读取频率和 PSD 数据（跳过第一行）
            data_lines = lines[1:]
            freqs, psds = [], []
            for line in data_lines:
                parts = line.split()
                if len(parts) < 2:  # 至少需要两列数据
                    continue
                try:
                    f = float(parts[0])  # 频率 (/μm)
                    p = float(parts[1])  # log10(PSD/nm⁴)
                    freqs.append(f)
                    psds.append(p)
                except ValueError:
                    # 跳过无法解析的行
                    continue

            if len(freqs) == 0:
                print(f"! 跳过 {filename}: 没有有效的数据行。")
                continue

            experimental_data.append((rms_val, freqs, psds))
            print(f"OK 已加载: {filename} → RMS = {rms_val:.2f} nm, {len(freqs)} 个数据点")

        except Exception as e:
            print(f"Error 读取 {filename} 时出错: {e}")

    if not experimental_data:
        raise ValueError("在 'data/' 目录中未找到有效的AFM数据文件。")

    # 按 RMS 值从小到大排序，便于后续插值
    experimental_data.sort(key=lambda x: x[0])
    print(f"总共加载了 {len(experimental_data)} 个不同RMS的PSD数据集（已去重）")
    return experimental_data

def generate_or_load_1d_psd(target_rms, scan_size_um, output_dir="result"):
    """
    Generate or load 1D PSD data, ensuring RMS value is close to target value.
    
    Parameters:
        target_rms: Target RMS value in nm
        output_dir: Output directory for generated PSD file
    
    Returns:
        tuple: (frequencies, PSD_amplitudes, target_rms)
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{scan_size_um:.0f}um_{target_rms:.1f}.txt"
    filepath = os.path.join(output_dir, filename)

    print(f"[PSD生成] 开始生成目标RMS={target_rms} nm的PSD")
    
    # 提取所有实验数据的RMS值、频率和PSD
    print("开始加载实验数据...")
    experimental_data = load_experimental_data_from_folder("data")
    psd_data_list = []
    for rms_val, freqs, psds in experimental_data:
        psd_data_list.append({
            'rms': rms_val,
            'freq': np.array(freqs),
            'amp': np.array(psds)
        })
    
    rms_values = np.array([data['rms'] for data in psd_data_list])
    
    # 统一频率网格
    all_freqs = []
    for data in psd_data_list:
        all_freqs.extend(data['freq'])
    common_freqs = np.sort(np.unique(all_freqs))
    print(f"[PSD生成] 使用所有数据频率的并集，共{len(common_freqs)}个频率点")

    # 将每个实验PSD插值到统一频率网格上
    amp_matrix = []
    for data in psd_data_list:
        interp_func = interp1d(
            data['freq'], data['amp'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        amp_interp = interp_func(common_freqs)
        amp_matrix.append(amp_interp)
    amp_matrix = np.array(amp_matrix)
    
    # 对每个频率点，使用GPR在RMS方向插值
    target_amp = []
    valid_freq_idx = []
    
    for freq_idx in range(len(common_freqs)):
        valid_mask = ~np.isnan(amp_matrix[:, freq_idx])
        valid_rms = rms_values[valid_mask]
        valid_amp = amp_matrix[valid_mask, freq_idx]
        
        if len(valid_rms) < 2:
            continue
        
        # 使用GPR进行插值
        kernel = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gpr.fit(valid_rms.reshape(-1, 1), valid_amp)
        amp_pred, _ = gpr.predict([[target_rms]], return_std=True)
        
        target_amp.append(amp_pred[0])
        valid_freq_idx.append(freq_idx)
    
    # 获取有效的频率和对应的PSD值
    valid_freq = common_freqs[valid_freq_idx]
    target_amp = np.array(target_amp)
    
    # 构建结果DataFrame并删除NaN值
    result_df = pd.DataFrame({'freq': valid_freq, 'amp': target_amp}).dropna()
    if len(result_df) == 0:
        raise ValueError("Failed to generate valid PSD (all points are NaN)")
    
    # 计算生成的PSD对应的RMS值
    calculated_rms = calculate_rms_from_psd_1d(result_df)
    
    # --- Rescale PSD by a constant factor so that RMS matches target exactly ---
    if calculated_rms > 0:
        scale = (target_rms / calculated_rms) ** 2
        result_df['amp'] = result_df['amp'] + np.log10(scale)
        calculated_rms_scaled = calculate_rms_from_psd_1d(result_df)
        print(f"[PSD Rescale] scale={scale:.6f}, RMS before rescale={calculated_rms:.6f} nm, RMS after rescale={calculated_rms_scaled:.6f} nm (target={target_rms:.6f} nm)")
        calculated_rms = calculated_rms_scaled
    
    # 保存数据
    with open(filepath, 'w') as f:
        f.write(f"RMS: {target_rms} nm\n")
        for _, row in result_df.iterrows():
            f.write(f"{row['freq']:.6e}\t{row['amp']:.6e}\n")

    print(f"[PSD Save] Interpolated PSD saved to: {filepath}")
    return result_df['freq'].values, result_df['amp'].values, target_rms

def generate_and_calibrate_psd(target_rms, scan_size_um, size, output_dir="result"):
    """
    Complete pipeline for generating a calibrated 1D PSD matches target RMS.
    
    This function:
    1. Generates/loads 1D PSD via GPR
    2. Generates initial surface to calculate actual RMS from synthesized topography
    3. Calibrates PSD using scale factor to match target RMS exactly
    4. Saves the calibrated PSD to files
    
    Returns:
        tuple: (freq_1d_um, psd_1d_calibrated, phase, height_initial)
    """
    from utils import (generate_symmetric_phase, 
                      generate_surface_from_psd_1d, 
                      calculate_rms_from_h)
    
    print("\n[PSD Calibration] Step 1: Generating 1D PSD with target RMS...")
    freq_1d_um, amp_1d, _ = generate_or_load_1d_psd(target_rms, scan_size_um, output_dir)
    psd_1d_df = pd.DataFrame({'freq': freq_1d_um, 'amp': amp_1d})
    
    print("[PSD Calibration] Step 2: Generating random phase and initial surface...")
    phase = generate_symmetric_phase(size, phase_type='gaussian')
    height_initial = generate_surface_from_psd_1d(
        psd_1d_df, scan_size_um=scan_size_um, size=size, phase=phase)
    
    rms_initial = calculate_rms_from_h(height_initial)
    print(f"  Generated surface initial RMS: {rms_initial:.3f} nm")
    
    print("[PSD Calibration] Step 3: Adjusting PSD to match target RMS...")
    scale_factor = target_rms / rms_initial
    print(f"  Scale factor: {scale_factor:.6f} (Target={target_rms:.3f}, Height={rms_initial:.3f})")
    
    # Scale 1D PSD (log10 scale)
    psd_1d_calibrated = amp_1d + 2.0 * np.log10(scale_factor)
    
    # Save calibrated 1D PSD
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"scaled_psd1d_uniform_rms_{target_rms:.1f}nm.txt")
    with open(out_file, 'w') as f:
        f.write("freq(/um)\tlog10(PSD/nm^4)\n")
        for freq, amp in zip(freq_1d_um, psd_1d_calibrated):
            f.write(f"{freq:.6e}\t{amp:.6e}\n")
    print(f"  Calibrated 1D PSD saved to: {out_file}")
    
    return freq_1d_um, psd_1d_calibrated, phase, height_initial

def save_psd_to_file(freqs, psds, filename):
    """Save PSD data to a file in NanoScope format."""
    with open(filename, 'w') as f:
        f.write("freq(/um)\tlog10(PSD/nm^4)\n")
        for freq, amp in zip(freqs, psds):
            f.write(f"{freq:.6e}\t{amp:.6e}\n")
