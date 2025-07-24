#!/usr/bin/env python3
import numpy as np

def split_dat_file(input_file, output_file, skip_cols):
    data = np.loadtxt(input_file, delimiter='\t')
    data_new = data[:, skip_cols:]
    np.savetxt(output_file, data_new, delimiter='\t', fmt='%.6f')
    print(f"已保存: {output_file}，去掉前{skip_cols}列，剩余{data_new.shape[1]}列")

if __name__ == "__main__":
    input_file = "run_50Hz_4p0.dat"
    split_dat_file(input_file, "run_50Hz_4p0_27.dat", 4)
    split_dat_file(input_file, "run_50Hz_4p0_24.dat", 7)
