import os
import numpy as np

def read_reference_frequencies(ref_file_path):
    """读取参考文件中的频率数据"""
    frequencies = []
    with open(ref_file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过第一行注释
    for line in lines[1:]:
        if line.strip():
            freq, _ = line.strip().split('\t')
            frequencies.append(float(freq))
    
    return frequencies

def read_psd_data(file_path):
    """读取PSD数据文件"""
    frequencies = []
    psd_values = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过第一行注释
    for line in lines[1:]:
        if line.strip():
            freq, psd = line.strip().split('\t')
            frequencies.append(float(freq))
            psd_values.append(float(psd))
    
    return frequencies, psd_values

def interpolate_psd(source_freqs, source_psd, target_freqs):
    """对PSD数据进行线性插值和外推"""
    # 使用numpy的interp函数，支持外推
    interpolated_psd = np.interp(target_freqs, source_freqs, source_psd)
    return interpolated_psd

def write_interpolated_data(file_path, header, target_freqs, interpolated_psd):
    """将插值后的数据写入文件"""
    with open(file_path, 'w') as f:
        f.write(header)
        for freq, psd in zip(target_freqs, interpolated_psd):
            # 使用科学计数法，保持与参考文件相同的格式
            f.write(f"{freq:.6e}\t{psd:.6e}\n")
    
    print(f"已写入插值后的数据到: {file_path}")

# 主函数
if __name__ == "__main__":
    # 文件路径
    ref_file = r"e:\premelting\RMS-PSD-data\实验\文献数据\演化前后PSD数据处理\30.0_00001.spm - NanoScope Analysis.txt"
    file1 = r"e:\premelting\RMS-PSD-data\实验\文献数据\演化前后PSD数据处理\0s.txt"
    file2 = r"e:\premelting\RMS-PSD-data\实验\文献数据\演化前后PSD数据处理\beforeRTA.txt"
    
    print("1. 读取参考文件中的频率数据...")
    target_freqs = read_reference_frequencies(ref_file)
    print(f"   参考文件频率点数量: {len(target_freqs)}")
    print(f"   频率范围: {min(target_freqs):.6e} 到 {max(target_freqs):.6e}")
    
    print("\n2. 处理0s.txt文件...")
    # 读取原文件的头信息和数据
    with open(file1, 'r') as f:
        header1 = f.readline()
    source_freqs1, source_psd1 = read_psd_data(file1)
    print(f"   原文件数据点数量: {len(source_freqs1)}")
    print(f"   原文件频率范围: {min(source_freqs1):.6e} 到 {max(source_freqs1):.6e}")
    
    # 插值
    interpolated_psd1 = interpolate_psd(source_freqs1, source_psd1, target_freqs)
    
    # 写回文件
    write_interpolated_data(file1, header1, target_freqs, interpolated_psd1)
    
    print("\n3. 处理beforeRTA.txt文件...")
    # 读取原文件的头信息和数据
    with open(file2, 'r') as f:
        header2 = f.readline()
    source_freqs2, source_psd2 = read_psd_data(file2)
    print(f"   原文件数据点数量: {len(source_freqs2)}")
    print(f"   原文件频率范围: {min(source_freqs2):.6e} 到 {max(source_freqs2):.6e}")
    
    # 插值
    interpolated_psd2 = interpolate_psd(source_freqs2, source_psd2, target_freqs)
    
    # 写回文件
    write_interpolated_data(file2, header2, target_freqs, interpolated_psd2)
    
    print("\n所有文件处理完成！")
