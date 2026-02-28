import argparse
import math
import numpy as np
#python  knowpsd-cal-rms2.py --height "/mnt/data/height-RTA-before-#23-center-10x10.0_00000.spm.txt" \   --psd   "/mnt/data/PSD-RTA-before-#23-center-10x10.0_00000.spm - NanoScope Analysis.txt"

def read_height_txt(path: str) -> np.ndarray:
    """
    读取 NanoScope 导出的 height 单列数据（可能带表头）。
    返回展平后的 float 数组（nm）。
    """
    # 尝试跳过第一行表头；如果失败就不跳
    try:
        arr = np.loadtxt(path, skiprows=1)
        if arr.size == 0:
            raise ValueError("empty after skiprows=1")
    except Exception:
        arr = np.loadtxt(path)

    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr


def rms_from_height(arr_nm: np.ndarray) -> float:
    """
    RMS = sqrt(mean((z-mean(z))^2)) = std(z)
    """
    z = arr_nm.astype(float)
    z = z - np.mean(z)
    return float(np.sqrt(np.mean(z * z)))


def read_psd_xz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 NanoScope PSD 导出的 XZ data，两列：
      x: k (1/um)
      y: log10(PSD) 或者 log10(isotropic PSD)（你这里是 log10(P)）
    文件可能第一行只有单位或表头，所以用更稳健的方式解析。
    """
    xs = []
    ys = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except Exception:
                continue
            xs.append(x)
            ys.append(y)

    if len(xs) < 5:
        raise ValueError(f"PSD file parsed too few numeric rows: {len(xs)}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # 按 x 排序，防止文件中出现乱序
    idx = np.argsort(x)
    return x[idx], y[idx]


def rms_from_psd_export(
    x_um_inv: np.ndarray,
    y_log10: np.ndarray,
    target_rms: float | None = None,
) -> tuple[float, dict]:
    """
    假设 y = log10(P), 其中 P = psd_radial/(2*pi*k)（与 cal_psd.py 的导出一致）。
    则 RMS^2 = (2*pi)^2 / 1e9 * ∫ (x^2 * 10^y) dx, 其中 x 单位 1/um。

    由于 NanoScope 的 x 可能是 bin 左边界或中心，这里尝试 3 种 shift：
      k_eff = x + shift, shift in {0, dk/2, dk}
    若给了 target_rms，就选最接近 target 的结果；否则默认用 dk/2。
    """
    x = x_um_inv.astype(float)
    y = y_log10.astype(float)

    # 判断 y 是否像 log10（通常在 0~20 左右）；不符合也照样按 log10 做
    P = np.power(10.0, y)

    # dk 用中位数估计
    diffs = np.diff(x)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("PSD x column has no positive spacing.")
    dk = float(np.median(diffs))

    shifts = [0.0, 0.5 * dk, 1.0 * dk]
    results = {}

    for shift in shifts:
        k_eff = x + shift
        integrand = (k_eff ** 2) * P
        I = float(np.trapz(integrand, x))  # dx 仍用导出 x 的间隔
        var = ((2.0 * math.pi) ** 2) / 1e9 * I
        rms = math.sqrt(var) if var > 0 else float("nan")
        results[shift] = {"dk": dk, "I": I, "var": var, "rms": rms}

    if target_rms is None:
        chosen_shift = 0.5 * dk
    else:
        chosen_shift = min(
            results.keys(),
            key=lambda s: abs(results[s]["rms"] - target_rms),
        )

    return results[chosen_shift]["rms"], {
        "chosen_shift": chosen_shift,
        "dk": results[chosen_shift]["dk"],
        "candidates": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=str, default=None, help="height txt path (single column, nm)")
    ap.add_argument("--psd", type=str, required=True, help="PSD XZ data txt path")
    args = ap.parse_args()

    # PSD -> RMS
    x, y = read_psd_xz(args.psd)

    rms_h = None
    if args.height is not None:
        h = read_height_txt(args.height)
        rms_h = rms_from_height(h)

    rms_p, info = rms_from_psd_export(x, y, target_rms=rms_h)

    print(f"[PSD ] RMS = {rms_p:.6f} nm")
    print(f"       chosen_shift = {info['chosen_shift']:.6g} (1/um), dk ~ {info['dk']:.6g} (1/um)")
    if rms_h is not None:
        print(f"[HGT ] RMS = {rms_h:.6f} nm")
        rel = (rms_p - rms_h) / rms_h * 100.0
        print(f"[DIFF] (PSD - HGT)/HGT = {rel:.3f}%")

        # 也把三个候选都打印出来，方便你核对 NanoScope 的 x 语义
        print("\nCandidates (shift -> RMS):")
        for s, d in info["candidates"].items():
            print(f"  shift={s:.6g}  rms={d['rms']:.6f}  var={d['var']:.6f}")


if __name__ == "__main__":
    main()
