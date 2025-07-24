#!/usr/bin/env python3
import numpy as np

def remove_columns(input_file, output_file, remove_idx):
    data = np.loadtxt(input_file, delimiter='\t')
    keep_idx = [i for i in range(data.shape[1]) if i not in remove_idx]
    data_new = data[:, keep_idx]
    np.savetxt(output_file, data_new, delimiter='\t', fmt='%.6f')
    print(f"已保存: {output_file}，去除列: {remove_idx}，剩余{data_new.shape[1]}列")

if __name__ == "__main__":
    # 第一组：去除第2、8、14、20列（索引1,7,13,19）
    remove_columns('run_50Hz_4p0_24.dat', 'run_50Hz_4p0_20_wo_frontal.dat', [1,7,13,19])
    # 第二组：去除第2、3、8、9、14、15、20、21列（索引1,2,7,8,13,14,19,20）
    remove_columns('run_50Hz_4p0_24.dat', 'run_50Hz_4p0_16_hip1.dat', [1,2,7,8,13,14,19,20])
