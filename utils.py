import numpy as np
import pandas as pd
def calculate_rms_from_h(h):
    """
    Calculate root mean square (RMS) roughness of a surface.
    
    Parameters:
        h: 2D array of surface heights in nm
    
    Returns:
        RMS value in nm
    """
    return np.sqrt(np.mean(h**2))

def calculate_rms_from_psd_1d(psd_df, freq_col='freq', amp_col='amp'):
    """
    Compute RMS directly from NanoScope exported XZ PSD data.

    Important: The XZ data exported from NanoScope's "2D ISO" PSD plot is *not*
    the raw 2D PSD density in k-space. Empirically (validated against the
    original height image), NanoScope's exported curve S(f) satisfies:

        RMS^2 = (2π)^2 ∫ k_eff(f)^2 * S(f) df

    where:
      - f is spatial frequency in 1/µm (as exported),
      - df is the (uniform) frequency bin width in 1/µm,
      - k_eff(f) = f + df  (NanoScope's effective radial bin convention),
      - S(f) = 10^{amp}  (because the plot uses Log scale).

    Unit conversion:
      - f in 1/µm → 1/nm by dividing by 1000
      - therefore (f^2 df) gains a factor 1/1000^3 = 1e-9

    So the implemented discrete approximation is:

        RMS^2 = (2π)^2 / 1e9 * ∫ (f + df)^2 * 10^{amp} df
    """
    freqs = psd_df[freq_col].to_numpy(dtype=float)
    amp_log = psd_df[amp_col].to_numpy(dtype=float)

    if freqs.size < 2:
        raise ValueError("PSD data has too few points to compute RMS.")

    # Convert Log(PSD) → PSD
    S = np.power(10.0, amp_log)

    # Frequency bin width (NanoScope exports a uniform grid for images)
    diffs = np.diff(freqs)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("PSD frequency column is not strictly increasing.")
    df = float(np.median(diffs))

    # NanoScope effective radial frequency convention (validated on your data)
    k_eff = freqs + df

    integral = np.trapz((k_eff ** 2) * S, freqs)
    var = ((2.0 * np.pi) ** 2) * integral / 1e9
    return float(np.sqrt(max(var, 0.0)))

def convert_1d_to_2d_psd(psd_1d_df, scan_size_um, size):
    """
    Convert NanoScope exported XZ PSD curve S(f) (from the "2D ISO" plot) into a 2D
    k-space PSD grid (shifted, zero-frequency at center) that is consistent with:

      1) calculate_rms_from_psd_1d() on the 1D exported curve, and
      2) calculate_rms_from_psd_2d() on the resulting 2D grid, and
      3) surface synthesis via IFFT in generate_surface_from_psd().

    Key point:
    The exported curve S(f) is not PSD_2D_iso(f) directly. From the RMS identity:

        RMS^2 = (2π)^2 ∫ k_eff^2 * S(f) df

    and the isotropic 2D PSD integral:

        RMS^2 = ∫∫ PSD_2D(kx,ky) dkx dky = 2π ∫ PSD_2D_iso(k) * k dk,

    we can construct a consistent isotropic 2D PSD density:

        PSD_2D_iso(k) = 2π * (k_eff(k)^2 / k) * S(k),   k>0
        with k_eff = k + dk   (dk is the frequency bin width)

    Here k and dk are in 1/nm. We implement this in nm units, then interpolate it
    onto the 2D FFT frequency grid (in 1/µm, later converted as needed).
    """
    freqs_um = psd_1d_df['freq'].to_numpy(dtype=float)
    amp_log = psd_1d_df['amp'].to_numpy(dtype=float)

    if freqs_um.size < 2:
        raise ValueError("1D PSD has too few points.")

    # Convert log → linear S(f)
    S = np.power(10.0, amp_log)

    # Frequency bin width in 1/µm  df
    diffs = np.diff(freqs_um)     #相邻差值
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("PSD frequency column is not strictly increasing.")
    df_um = float(np.median(diffs))    #中位数

    # Work in 1/nm for a clean variance identity
    freqs_nm = freqs_um / 1000.0
    df_nm = df_um / 1000.0

    # Build isotropic 2D PSD density in k-space (nm^4)
    # PSD2D_iso(k) = 2π * ((k+dk)^2 / k) * S(k), k>0
    k = freqs_nm
    k_eff = k + df_nm
    psd2d_iso = 2.0 * np.pi * (k_eff ** 2) / np.maximum(k, 1e-30) * S

    # 2D frequency grid (fftshifted) in 1/µm
    dx_um = scan_size_um / size
    k1_um = np.fft.fftfreq(size, d=dx_um)
    k1_um = np.fft.fftshift(k1_um)
    kx_um, ky_um = np.meshgrid(k1_um, k1_um, indexing='xy')
    k_mag_um = np.sqrt(kx_um ** 2 + ky_um ** 2)
    #print(f"dx_um: {dx_um}")
    #print(f"k1_um: {k1_um}")
    #print(f"k_mag_um: {k_mag_um}")
    #print(f"kx_um: {kx_um}")
    #print(f"ky_um: {ky_um}")    
    #k_mag_um =  np.sqrt(kx_um ** 2 )

    # Interpolate in 1/nm domain
    from scipy.interpolate import interp1d
    interp_func = interp1d(
        freqs_nm,
        psd2d_iso,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    psd_2d = interp_func(k_mag_um /1000)
    #psd_2d = interp_func(k_mag_um )

    # Remove DC
    #psd_2d[k_mag_um < 1e-12] = 0.0

    # Useful debug prints
    valid = k_mag_um > 1e-12
    print(f"扫描尺寸: {scan_size_um} μm × {scan_size_um} μm, 像素数: {size}×{size}, 采样间隔dx: {dx_um:.6f} μm")
    print(f"  Input data freq range: {freqs_um.min():.2e} ~ {freqs_um.max():.2e} /μm")
    print(f"  2D grid freq range: {k_mag_um[valid].min():.2e} ~ {k_mag_um[valid].max():.2e} /μm")
    print(f"  2D PSD statistics: min={psd_2d[valid].min():.2e}, max={psd_2d[valid].max():.2e}, mean={psd_2d[valid].mean():.2e}")

    return kx_um, ky_um, k_mag_um, psd_2d

def generate_symmetric_phase(size, phase_type='uniform'):
    """
    Generate a Hermitian-symmetric phase matrix in **unshifted FFT order**.

    This is designed to be used together with:
      H = amp_unshifted * exp(1j * phase_unshifted)
      h = ifft2(H).real

    Hermitian symmetry needed for real-valued spatial height:
      phase(-k) = -phase(k)
    For self-conjugate points (DC and Nyquist lines), phase must be 0 or π.
    """
    np.random.seed(42)  # Fixed seed for reproducibility
    phase = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            i2 = (-i) % size
            j2 = (-j) % size

            # Only fill one side of each conjugate pair
            if (i > i2) or (i == i2 and j > j2):
                continue

            if i == i2 and j == j2:
                # Self-conjugate point -> real coefficient
                phase[i, j] = 0.0 if np.random.random() < 0.5 else np.pi
            else:
                if phase_type == 'uniform':
                    ang = np.random.uniform(0.0, 2.0 * np.pi)
                elif phase_type == 'gaussian':
                    ang = np.random.normal(0, np.pi)
                else:
                    ang = np.random.uniform(0.0, 2.0 * np.pi)
                phase[i, j] = ang
                phase[i2, j2] = -ang

    phase[0, 0] = 0.0
    return phase

def generate_surface_from_psd_2d(psd_2d, scan_size_um, size=None, seed=None, phase=None, phase_type='gaussian'):
    """
    Generate surface height distribution from 2D PSD, based on psd_1d2d_fft_rms.py implementation.
    
    Parameters:
        psd_2d: 2D PSD array (nm^4 units)
        scan_size_um: Scan size in micrometers
        size: Size of the surface grid
        seed: Optional random seed for phase generation
        phase: Optional phase array (to ensure consistent evolution)
        phase_type: Type of phase distribution if generating new phase
    
    Returns:
        2D array: Surface height distribution in nm
    """
    if seed is not None:
        np.random.seed(seed)
    
    if size is None:
        size = psd_2d.shape[0]
    
    # === 1. Calculate spatial sampling interval (nm) ===
    L_nm = scan_size_um * 1000           # Scan size (nm)
    dx_nm = L_nm / size # Spatial sampling interval (nm)
    dk_nm = 1.0 / (size * dx_nm)  # Wave number interval (1/nm)  f=1/L
    
    # === 2. Calculate DFT coefficient amplitude ===
    # |H(k)| = sqrt(PSD(k) * dk_x * dk_y) (unit: nm)
    amp_fourier = np.sqrt(psd_2d * dk_nm**2)  # Unit: nm

    # === 3. Use or generate phase ===
    # If external phase is provided, use it; otherwise generate new symmetric phase
    if phase is None:
        # Generate symmetric phase to ensure real-valued result
        phase = generate_symmetric_phase(size, phase_type)

    # === 4. Build complex amplitude ===
    # Complex amplitude = amplitude * exp(j*phase)
    complex_amp = amp_fourier * np.exp(1j * phase)

    # === 5. Apply IFFT shift for proper inverse transform ===
    # Shift to DFT order required by ifft2 (DC at [0,0])
    complex_dft_order = np.fft.ifftshift(complex_amp)

    # === 6. Inverse transform to get height ===
    h = np.fft.ifft2(complex_dft_order).real  # Unit: nm
    
    # === 7. Apply scaling factor for discrete FFT amplitude correction ===
    # Scale by size^2 to correct for FFT scaling (as in psd_1d2d_fft_rms.py)
    h *= (size**2) # 更清晰地写出归一化因子
    
    return h

def generate_surface_from_psd_1d(psd_1d_df, scan_size_um, size, seed=None, phase=None, phase_type='gaussian'):
    kx_um, ky_um, k_mag_um, psd_2d = convert_1d_to_2d_psd(psd_1d_df, scan_size_um, size)
    h = generate_surface_from_psd_2d(
        psd_2d,
        scan_size_um=scan_size_um,
        size=size,
        seed=seed,
        phase=phase,
        phase_type=phase_type
    )
    return h

def compute_temperature_profile(T0, anneal_temp, anneal_time, heating_rate, cooling_rate):
    """
    Compute complete temperature-time profile: heating → holding → cooling.
    
    Parameters:
        T0: Initial temperature in K
        anneal_temp: Annealing temperature in K
        anneal_time: Holding time at annealing temperature in seconds
        heating_rate: Heating rate in K/s
        cooling_rate: Cooling rate in K/s
    
    Returns:
        tuple: (time_points, temperature_points) - arrays of time and corresponding temperature
    """
    if anneal_temp <= T0:
        raise ValueError("退火温度必须大于初始温度")
    
    # 计算升温和降温时间
    heating_time = (anneal_temp - T0) / heating_rate
    cooling_time = (anneal_temp - T0) / cooling_rate
    total_time = heating_time + anneal_time + cooling_time

    # 生成各阶段的时间点
    # 升温：包含终点
    t_heating = np.linspace(0, heating_time, 50, endpoint=True)  # [0, ..., heating_time]
    
    # 保温：排除起点（避免重复），但包含后续点
    if anneal_time > 0:
        t_anneal = np.linspace(heating_time, heating_time + anneal_time, 101, endpoint=True)[1:]
    else:
        t_anneal = np.array([])
    
    # 降温：排除起点（避免重复）
    t_cooling = np.linspace(heating_time + anneal_time, total_time, 51, endpoint=True)[1:]
    
    t_all = np.concatenate([t_heating, t_anneal, t_cooling])

    # 计算各阶段的温度
    T_heating = T0 + heating_rate * t_heating      # 线性升温
    T_anneal = np.full_like(t_anneal, anneal_temp)  # 恒温
    T_cooling = anneal_temp - cooling_rate * (t_cooling - (heating_time + anneal_time))  # 线性降温
    T_all = np.concatenate([T_heating, T_anneal, T_cooling])
    
    return t_all, T_all

def visualize_evolution_results(t_all, T_all, freq_1d_um, psd_1d_initial, psd_1d_evolved, 
                                height_initial, height_evolved, config, output_path='result.png'):
    """
    Generate a comprehensive visualization image for the simulation results.
    """
    import matplotlib.pyplot as plt
    import os

    target_rms = config.get('target_rms')
    T_anneal = config.get('T_anneal')
    anneal_time = config.get('anneal_time')
    heating_rate = config.get('heating_rate')
    cooling_rate = config.get('cooling_rate')
    T_initial = config.get('T_initial', 300)
    scan_size_um = config.get('scan_size_um')

    rms_initial = calculate_rms_from_h(height_initial)
    rms_evolved = calculate_rms_from_h(height_evolved)
    rms_reduction = rms_initial - rms_evolved
    reduction_percent = (rms_reduction / rms_initial) * 100 if rms_initial > 0 else 0

    freq_min = 1.0 / scan_size_um

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # 1. Temperature profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_all, T_all, 'r-', linewidth=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title(f'Annealing Profile (Temp: {T_anneal}K, Time: {anneal_time}s)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    heating_end = (T_anneal - T_initial) / heating_rate
    cooling_start = heating_end + anneal_time
    ax1.axvline(x=heating_end, color='orange', linestyle='--', alpha=0.7, label='Heating End')
    ax1.axvline(x=cooling_start, color='purple', linestyle='--', alpha=0.7, label='Cooling Start')
    ax1.legend()

    # 2. PSD comparison
    ax2 = fig.add_subplot(gs[0, 1])
    # psd_1d_initial and psd_1d_evolved are expected to be log10 values
    ax2.loglog(freq_1d_um, 10**psd_1d_initial, 'k-', label="Initial", linewidth=2)
    ax2.loglog(freq_1d_um, 10**psd_1d_evolved, 'r-', label="Evolved", linewidth=2)
    ax2.set_xlabel('Spatial Frequency (/μm)')
    ax2.set_ylabel('PSD (nm⁴)')
    ax2.set_title('Power Spectral Density Comparison')
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    # 3. Initial surface
    ax3 = fig.add_subplot(gs[1, 0])
    vmin = min(np.nanmin(height_initial), np.nanmin(height_evolved))
    vmax = max(np.nanmax(height_initial), np.nanmax(height_evolved))
    im1 = ax3.imshow(height_initial, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Initial Surface\nRMS={rms_initial:.6f} nm', fontsize=14)
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04).set_label('Height (nm)')

    # 4. Evolved surface
    ax4 = fig.add_subplot(gs[1, 1])
    im2 = ax4.imshow(height_evolved, cmap='viridis', vmin=vmin, vmax=vmax)
    ax4.set_title(f'Evolved Surface\nRMS={rms_evolved:.6f} nm', fontsize=14)
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04).set_label('Height (nm)')

    # 5. Surface difference
    ax5 = fig.add_subplot(gs[2, 0])
    diff_surface = height_evolved - height_initial
    im3 = ax5.imshow(diff_surface, cmap='RdBu_r')
    ax5.set_title('Surface Difference\n(Evolved - Initial)', fontsize=14)
    plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04).set_label('Height Difference (nm)')

    # 6. Table of results
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    params = [
        ['Initial RMS', f'{rms_initial:.6f} nm'],
        ['Evolved RMS', f'{rms_evolved:.6f} nm'],
        ['Roughness Reduction', f'{rms_reduction:.6f} nm ({reduction_percent:.1f}%)'],
        ['Heat Profile', f'{T_initial}K → {T_anneal}K'],
        ['Hold Time', f'{anneal_time} s'],
        ['Noise c1', f'{config.get("c1", "N/A"):.2e}'],
        ['Noise c2', f'{config.get("c2", "N/A"):.2e}'],
        ['Total Time', f'{t_all[-1]:.2f} s'],
        ['Heating Rate', f'{heating_rate} K/s'],
        ['Cooling Rate', f'{cooling_rate} K/s'],
        ['Min Frequency', f'{freq_min:.2f}/μm'],
        ['Corresponding Wavelength', f'{1/freq_min:.2f}μm']
    ]
    table = ax6.table(cellText=params, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == 0:
            cell.set_text_props(weight='bold')
            if i == 0: cell.set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Result image saved as '{output_path}'")
