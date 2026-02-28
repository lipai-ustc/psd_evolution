import os
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

class EV:
    def __init__(self):
        self.datasets = [
            # 每个元组包含: (温度列表, K值列表, 波长标签)
            # K: 衰减常数 (1/s), λ: 波长 (nm)
            ([1300, 1400, 1500, 1600, 1700, 1750, 1800],
            [2.428085e7, 4.846299e7, 1.201836e8, 2.845039e8, 1.038310e9, 1.881473e9, 7.163844e9], 'L=10 nm'),
            ([1400, 1500, 1600, 1700, 1750, 1800, 1900],
            [1.07217e6, 2.981999e7, 1.625615e7, 4.986440e7, 2.620726e8, 5.042448e8, 6.265459e10], 'L=20 nm'),
            ([1500, 1600, 1700, 1750, 1800, 1900],
            [1.344085e6, 1.411912e6, 1.034719e7, 2.497419e7, 1.71299e8, 1.184389e10], 'L=30 nm'),
            ([1600, 1700, 1750, 1800, 1900],
            [2.197188e6, 8.242978e6, 2.191620e7, 7.307947e7, 9.059217e9], 'L=40 nm'),
            ([1600, 1700, 1750, 1800, 1900],
            [4.652165e5, 2.491983e6, 1.478187e7, 2.497893e7, 1.425268e9], 'L=50 nm')
        ]
        # build or load cached interpolator (call with force_rebuild=True to refresh)
        self.get_vprime = self.build_vprime_interpolator()

    def _make_vprime_func(self, params):
        """Reconstruct v_prime_func from saved parameter dictionary.

        The params dict must contain all the fitted coefficients and constants
        that describe the piecewise models as computed in the original build.
        """
        popt_linear = params['popt_linear']
        popt_quadratic = params['popt_quadratic']
        popt_linear3 = params['popt_linear3']
        log_v_1500 = params['log_v_1500']
        invT_1500 = params['invT_1500']
        log_v_1800 = params['log_v_1800']
        invT_1800 = params['invT_1800']
        log_vprime_shift = params.get('log_vprime_shift', 0.0)

        # define the primitive model functions so they can be reused
        def linear_func_log(invT, a, b):
            return a * invT + b

        def quadratic_model(x, c, d):
            return log_v_1500 + c * (x - invT_1500) + d * (x - invT_1500)**2

        def linear_func_log3(x, e):
            return log_v_1800 + e * (x - invT_1800)

        def v_prime_func(T):
            is_scalar = np.isscalar(T)
            if is_scalar:
                T = np.array([T])
            else:
                T = np.asarray(T)
            result = np.zeros_like(T, dtype=np.float64)
            for i, t in enumerate(T):
                invT = 1.0 / t
                if t < 1300:
                    log_v_fit = linear_func_log(invT, *popt_linear)
                elif t <= 1500:
                    log_v_fit = linear_func_log(invT, *popt_linear)
                elif t <= 1800:
                    log_v_fit = quadratic_model(invT, *popt_quadratic)
                else:
                    log_v_fit = linear_func_log3(invT, *popt_linear3)
                result[i] = 10**(log_v_fit + log_vprime_shift)
            return result[0] if is_scalar else result
        return v_prime_func

    def build_vprime_interpolator(self, force_rebuild: bool = False):
        """
        Build interpolator function for v'(T) to calculate v' values based on temperature.
        Implementation follows segmented fitting strategy:
        1. Below 1300K: Extrapolation using linear fit from 1300K-1500K
        2. 1300K-1500K: Linear fit on log(v') vs 1/T
        3. 1500K-1900K: Best fit model (from linear, quadratic, exponential) on log(v') vs 1/T

        If a cached parameter file exists (``vprime_params.pkl``) and
        ``force_rebuild`` is False, the function will be reconstructed from
        those parameters instead of re-running the full fitting procedure.

        Args:
            force_rebuild (bool): if True, ignore any cache and recompute.

        Returns:
            function: Interpolation function that maps temperature to v' value
        """
        # try loading cached parameters to avoid recomputation
        cache_file = 'vprime_params.pkl'
        if not force_rebuild and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    params = pickle.load(f)
                # rebuild function from saved params
                return self._make_vprime_func(params)
            except Exception:
                # if loading fails, proceed to rebuild and overwrite cache
                pass

        # 收集所有数据
        all_temperatures = []
        all_vprime_values = []
        all_L = []
        
        for temps, k_vals, label in self.datasets:
            lambda_nm = float(label.split('=')[1].split()[0])
            factor = lambda_nm ** 4  # 根据用户要求，使用 v' = K * λ^4 公式
            
            for T, K in zip(temps, k_vals):
                vprime = K * factor  # v'单位: nm⁴/s
                all_temperatures.append(T)
                all_vprime_values.append(vprime)
                all_L.append(lambda_nm)
        
        all_temperatures = np.array(all_temperatures)
        all_vprime_values = np.array(all_vprime_values)
        all_L = np.array(all_L)
        
        # 按温度分组并计算加权平均v'值（L值越大权重越大）
        temperature_groups = {}
        for T, v, L in zip(all_temperatures, all_vprime_values, all_L):
            if T not in temperature_groups:
                temperature_groups[T] = []
            temperature_groups[T].append((v, L))
        
        # 计算每个温度点的加权平均v'值
        T_list = []
        vprime_avg_list = []
        invT_list = []
        
        for T, data in sorted(temperature_groups.items()):
            v_values = np.array([d[0] for d in data])
            L_values = np.array([d[1] for d in data])
            
            # 计算权重：L值越大权重越大，归一化
            weights = L_values / np.max(L_values)
            weights = weights / np.sum(weights)  # 归一化
            
            # 计算加权平均v'值
            vprime_avg = np.sum(v_values * weights)
            
            T_list.append(float(T))
            vprime_avg_list.append(vprime_avg)
            invT_list.append(1.0 / float(T))
        
        T_array = np.array(T_list)
        vprime_avg_array = np.array(vprime_avg_list)
        invT_array = np.array(invT_list)
        
        # 第一段：1300K-1500K 线性拟合（对log(v')）
        T1_min = 1300
        T1_max = 1500
        mask1 = (T_array >= T1_min) & (T_array <= T1_max)
        
        if np.sum(mask1) < 2:
            raise ValueError("第一段拟合数据不足")
        
        invT1 = invT_array[mask1]
        v1 = vprime_avg_array[mask1]
        log_v1 = np.log10(v1)
        
        # 线性拟合函数（对log(v')）
        def linear_func_log(invT, a, b):
            return a * invT + b
        
        # 进行线性拟合
        popt_linear, pcov_linear = curve_fit(linear_func_log, invT1, log_v1, maxfev=10000)
        
        # 第二段：1500K-1800K 二次拟合（对log(v')）
        T2_min = 1500
        T2_max = 1800
        mask2 = (T_array >= T2_min) & (T_array <= T2_max)
        
        if np.sum(mask2) < 3:
            raise ValueError("第二段拟合数据不足")
        
        invT2 = invT_array[mask2]
        v2 = vprime_avg_array[mask2]
        log_v2 = np.log10(v2)
        
        # 计算1500K处的1/T值和log(v')值（确保连续性）
        invT_1500 = 1.0 / 1500
        log_v_1500 = linear_func_log(invT_1500, *popt_linear)
        v_1500 = 10**log_v_1500
        
        # 二次拟合函数（对log(v')，确保在1500K处连续）
        def quadratic_model(x, c, d):
            return log_v_1500 + c * (x - invT_1500) + d * (x - invT_1500)**2
        
        # 进行二次拟合
        try:
            popt_quadratic, pcov_quadratic = curve_fit(quadratic_model, invT2, log_v2, p0=(1.0, 1.0), maxfev=10000)
        except Exception as e:
            print(f"拟合quadratic模型时出错: {e}")
            raise ValueError("无法进行quadratic拟合")
        
        # 第三段：1800K-1900K 线性拟合（对log(v')）
        T3_min = 1800
        T3_max = 1900
        mask3 = (T_array >= T3_min) & (T_array <= T3_max)
        
        if np.sum(mask3) < 2:
            raise ValueError("第三段拟合数据不足")
        
        invT3 = invT_array[mask3]
        v3 = vprime_avg_array[mask3]
        log_v3 = np.log10(v3)
        
        # 计算1800K处的1/T值和log(v')值（确保连续性）
        invT_1800 = 1.0 / 1800
        log_v_1800 = quadratic_model(invT_1800, *popt_quadratic)
        v_1800 = 10**log_v_1800
        
        # 线性拟合函数（对log(v')，确保在1800K处连续）
        def linear_func_log3(x, e):
            return log_v_1800 + e * (x - invT_1800)
        
        # 进行线性拟合
        try:
            popt_linear3, pcov_linear3 = curve_fit(linear_func_log3, invT3, log_v3, p0=(1.0,), maxfev=10000)
        except Exception as e:
            print(f"拟合第三段线性模型时出错: {e}")
            raise ValueError("无法进行第三段线性拟合")
        
        # 定义v'计算函数
        def v_prime_func(T):
            # 检查是否为标量
            is_scalar = np.isscalar(T)
            if is_scalar:
                T = np.array([T])
            else:
                T = np.asarray(T)
            
            result = np.zeros_like(T, dtype=np.float64)
            
            # 逐个处理每个温度点
            for i, t in enumerate(T):
                # 1. 低于1300K：使用1300K-1500K的拟合曲线进行外推
                if t < 1300:
                    invT = 1.0 / t
                    log_v_fit = linear_func_log(invT, *popt_linear)
                    result[i] = 10**log_v_fit
                # 2. 1300K-1500K：使用线性拟合
                elif 1300 <= t <= 1500:
                    invT = 1.0 / t
                    log_v_fit = linear_func_log(invT, *popt_linear)
                    result[i] = 10**log_v_fit
                # 3. 1500K-1800K：使用二次拟合
                elif 1500 < t <= 1800:
                    invT = 1.0 / t
                    log_v_fit = quadratic_model(invT, *popt_quadratic)
                    result[i] = 10**log_v_fit
                # 4. 1800K以上：使用第三段线性拟合
                else:
                    invT = 1.0 / t
                    log_v_fit = linear_func_log3(invT, *popt_linear3)
                    result[i] = 10**log_v_fit
            
            # 返回结果，保持与输入相同的类型
            return result[0] if is_scalar else result
        
        # 输出拟合结果信息
        print(f"=== v'分段拟合结果 ===")
        print(f"1. 1300K-1500K 线性拟合（对log(v')）:")
        print(f"   方程: log10(v') = {popt_linear[0]:.2e} * (1/T) + {popt_linear[1]:.2e}")
        print(f"   v' = 10^({popt_linear[0]:.2e} * (1/T) + {popt_linear[1]:.2e})")
        
        print(f"\n2. 1500K-1800K 二次拟合（对log(v')）:")
        c, d = popt_quadratic
        print(f"   方程: log10(v') = {log_v_1500:.2f} + {c:.2e}*(1/T - {invT_1500:.6f}) + {d:.2e}*(1/T - {invT_1500:.6f})²")
        print(f"   v' = 10^({log_v_1500:.2f} + {c:.2e}*(1/T - {invT_1500:.6f}) + {d:.2e}*(1/T - {invT_1500:.6f})²)")
        
        print(f"\n3. 1800K-1900K 线性拟合（对log(v')）:")
        e = popt_linear3[0]
        print(f"   方程: log10(v') = {log_v_1800:.2f} + {e:.2e}*(1/T - {invT_1800:.6f})")
        print(f"   v' = 10^({log_v_1800:.2f} + {e:.2e}*(1/T - {invT_1800:.6f}))")
        
        print(f"\n4. 连续性检查:")
        print(f"   1500K处: v' = {v_1500:.3e} nm⁴/s")
        print(f"   1800K处: v' = {v_1800:.3e} nm⁴/s")
            # =============================================================================
        # 平移设置 - 通过注释开关是否启用平移
        # =============================================================================
        # 如果需要启用平移，将下一行的注释符号(#)去掉，并将下面的USE_SHIFT False改为True
        USE_SHIFT = True
        #USE_SHIFT = False  # 改为True以启用平移功能
        # 默认无平移
        log_vprime_shift = 0.0
        
        if USE_SHIFT:
            # 目标温度和目标v'值（用于对齐）
            target_T = 1608.35  # 目标温度 (K)
            target_vprime = 191739828299.75107  # 目标v'值 (nm⁴/s)
            
            # 计算当前拟合曲线在target_T处的v'值
            target_invT = 1.0 / target_T
            
            # 判断目标温度所在的温度段
            if target_T <= 1500:
                current_log_v = linear_func_log(target_invT, *popt_linear)
            elif target_T <= 1800:
                current_log_v = quadratic_model(target_invT, *popt_quadratic)
            else:
                current_log_v = linear_func_log3(target_invT, *popt_linear3)
            
            current_vprime = 10**current_log_v
            target_log_v = np.log10(target_vprime)
            
            # 计算log(v')的平移量
            log_vprime_shift = target_log_v - current_log_v
            
            print(f"\n=== 平移计算 (已启用) ===")
            print(f"目标温度: {target_T} K")
            print(f"目标v'值: {target_vprime:.6e} nm⁴/s")
            print(f"当前拟合v'值 (在{target_T}K): {current_vprime:.6e} nm⁴/s")
            print(f"log10(v')的平移量: {log_vprime_shift:.6f}")
            print(f"平移后v'值 = 当前v'值 × 10^{log_vprime_shift:.6f}")
            
            # 创建平移后的v'计算函数（保持返回名称为v_prime_func）
            def v_prime_func(T):
                # 检查是否为标量
                is_scalar = np.isscalar(T)
                if is_scalar:
                    T = np.array([T])
                else:
                    T = np.asarray(T)
                
                result = np.zeros_like(T, dtype=np.float64)
                
                # 逐个处理每个温度点
                for i, t in enumerate(T):
                    # 1. 低于1300K：使用1300K-1500K的拟合曲线进行外推
                    if t < 1300:
                        invT = 1.0 / t
                        log_v_fit = linear_func_log(invT, *popt_linear)
                        result[i] = 10**(log_v_fit + log_vprime_shift)
                    # 2. 1300K-1500K：使用线性拟合
                    elif 1300 <= t <= 1500:
                        invT = 1.0 / t
                        log_v_fit = linear_func_log(invT, *popt_linear)
                        result[i] = 10**(log_v_fit + log_vprime_shift)
                    # 3. 1500K-1800K：使用二次拟合
                    elif 1500 < t <= 1800:
                        invT = 1.0 / t
                        log_v_fit = quadratic_model(invT, *popt_quadratic)
                        result[i] = 10**(log_v_fit + log_vprime_shift)
                    # 4. 1800K以上：使用第三段线性拟合
                    else:
                        invT = 1.0 / t
                        log_v_fit = linear_func_log3(invT, *popt_linear3)
                        result[i] = 10**(log_v_fit + log_vprime_shift)
                
                # 返回结果，保持与输入相同的类型
                return result[0] if is_scalar else result

        # 缓存当前的参数（包括shift）
        params = {
            'popt_linear': popt_linear,
            'popt_quadratic': popt_quadratic,
            'popt_linear3': popt_linear3,
            'log_v_1500': log_v_1500,
            'invT_1500': invT_1500,
            'log_v_1800': log_v_1800,
            'invT_1800': invT_1800,
            'log_vprime_shift': log_vprime_shift
        }
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(params, f)
        except Exception as e:
            print(f"Warning: could not write v' cache: {e}")

        # 如果USE_SHIFT为False，保留原始的v_prime_func（已在上面定义）
        
        return v_prime_func

    def evolve_psd_1d(self, psd_1d, freq_1d_um, temperature_profile, time_points, dt_max, c1, c2):
        """
        Evolve 1D PSD using the exact per-step solution of the linear M-H equation
        with time-dependent temperature T(t).

        Parameters
        ----------
        psd_1d : ndarray or DataFrame
            1D PSD values (linear scale, nm⁴). If DataFrame, 'amp' column is used.
        freq_1d_um : ndarray
            1D array of spatial frequencies (/μm).
        temperature_profile : ndarray
            Temperature values at time points (K).
        time_points : ndarray
            Time points (s).
        dt_max : float
            Maximum time step size (s).
        c1 : float
            Parameterized coefficient for non-conservative noise (delta = c1 * v' * T).
        c2 : float
            Parameterized coefficient for conservative noise (gamma = c2 * v' * T).

        Returns
        -------
        psd_evolved : ndarray
            Evolved 1D PSD array (same shape as input).
        """
        if isinstance(psd_1d, pd.DataFrame):
            psd_current = psd_1d['amp'].values.copy()
        else:
            psd_current = np.asarray(psd_1d).copy()

        # freq_1d is in /μm → convert to /nm
        f = np.maximum(freq_1d_um / 1000.0, 1e-30)

        # build fine time grid
        total_time = time_points[-1] - time_points[0]
        n_steps = max(1, int(np.ceil(total_time / dt_max)))
        t_eval = np.linspace(time_points[0], time_points[-1], n_steps + 1)

        T_interp = interp1d(time_points, temperature_profile, kind='linear')
        T_eval = T_interp(t_eval)

        for i in range(n_steps):
            dt = t_eval[i + 1] - t_eval[i]
            T_mid = 0.5 * (T_eval[i] + T_eval[i + 1])

            v_prime = self.get_vprime(T_mid)  # nm^4 / s
            
            # Parameterized noise: delta = c1*v'*T, gamma = c2*v'*T
            delta_t = c1 * v_prime * T_mid
            gamma_t = c2 * v_prime * T_mid

            # decay factor (broadcasted over frequency array)
            A = np.exp(-2.0 * v_prime * f**4 * dt)

            # noise injection term: [delta_t + gamma_t*f^2] / (2pi)^4 * (1-A)/(2v'f^4)
            # IMPORTANT: For 1D evolution to match 2D isotropic evolution, we divide by (2*pi*f)
            B_coeff = (delta_t + gamma_t * f**2) / (((2.0 * np.pi)**4) * (2.0 * np.pi * f))
            denom = 2.0 * v_prime * f**4
            
            noise_term = np.zeros_like(A)
            mask = denom > 1e-60
            noise_term[mask] = B_coeff[mask] * (1.0 - A[mask]) / denom[mask]
            noise_term[~mask] = B_coeff[~mask] * dt

            psd_current = psd_current * A + noise_term

        return psd_current
