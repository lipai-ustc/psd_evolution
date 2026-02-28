import os

def sort_psd_file(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 提取第一行注释
    header = lines[0]
    
    # 提取数据行并转换为数字
    data = []
    for line in lines[1:]:
        if line.strip():
            freq, psd = line.strip().split('\t')
            data.append((float(freq), float(psd)))
    
    # 按第一列升序排序
    sorted_data = sorted(data, key=lambda x: x[0])
    
    # 写回原文件
    with open(file_path, 'w') as f:
        f.write(header)
        for freq, psd in sorted_data:
            f.write(f"{freq}\t{psd}\n")
    
    print(f"已处理文件: {file_path}")
    print(f"排序前数据行数: {len(data)}")
    print(f"排序后数据行数: {len(sorted_data)}")
    print()

# 主函数
if __name__ == "__main__":
    # 获取当前目录下的所有txt文件
    current_dir = os.getcwd()
    txt_files = [f for f in os.listdir(current_dir) if f.endswith('.txt')]
    
    print(f"找到 {len(txt_files)} 个txt文件需要处理")
    print()
    
    # 处理每个文件
    for txt_file in txt_files:
        file_path = os.path.join(current_dir, txt_file)
        sort_psd_file(file_path)
    
    print("所有文件处理完成！")
