#!/usr/bin/env python3
"""
脚本用于处理轨迹文件：将第11列和第17列取负值
"""

import numpy as np

def process_trajectory_file(input_file, output_file):
    """
    读取轨迹文件，将第11列和第17列取负值，保存为新文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 读取数据
    data = np.loadtxt(input_file, delimiter='\t')
    
    print(f"原始数据形状: {data.shape}")
    print(f"数据列数: {data.shape[1]}")
    
    # 膝关节正方向定义相反
    # 将第11列,第17列,第23列和第29列取负值 (注意：Python索引从0开始，所以是10和16, 22和28)
    data[:, 10] = -data[:, 10]  # 第11列
    data[:, 16] = -data[:, 16]  # 第17列
    data[:, 22] = -data[:, 22]  # 第23列
    data[:, 28] = -data[:, 28]  # 第29列

    # 原始数据中第8列为frontal, 第9列为sagittal. hhfc中第8列为sagittal, 第9列为frontal
    # 交换第8列和第9列, 第20列和第21列 (Python索引：7和8, 19和20)
    temp_col = data[:, 7].copy()  # 临时保存第8列
    data[:, 7] = data[:, 8]       # 第8列 = 第9列
    data[:, 8] = temp_col         # 第9列 = 原第8列
    temp_col = data[:, 19].copy()  # 临时保存第20列
    data[:, 19] = data[:, 20]      # 第20列 = 第21列
    data[:, 20] = temp_col          # 第21列 = 原第20列

    # 原始数据中第14列为frontal, 第15列为sagittal. hhfc中第14列为sagittal, 第15列为frontal
    # 交换第14列和第15列, 第26列和第27列 (Python索引：13和14, 25和26)
    temp_col = data[:, 13].copy() # 临时保存第14列
    data[:, 13] = data[:, 14]     # 第14列 = 第15列
    data[:, 14] = temp_col        # 第15列 = 原第14列
    temp_col = data[:, 25].copy() # 临时保存第26列
    data[:, 25] = data[:, 26]     # 第26列 = 第27列
    data[:, 26] = temp_col        # 第27列 = 原第26列

    # saggital(hip_pitch)的正方向定义相反
    # 将第8列, 第14列, 第20列和第26列取负值 (Python索引：7, 13, 19和25)
    data[:, 7] = -data[:, 7]
    data[:, 13] = -data[:, 13]
    data[:, 19] = -data[:, 19]
    data[:, 25] = -data[:, 25]

    print(f"已将膝关节角度及角速度反向")
    print(f"已交换髋关节saggital与frontal角度与角速度")
    print(f"已将髋关节saggital角度及角速度反向")
    
    # 保存到新文件，使用制表符分隔
    np.savetxt(output_file, data, delimiter='\t', fmt='%.6f')
    
    print(f"处理完成，保存到: {output_file}")

if __name__ == "__main__":
    input_file = "/home/harmony/legged_gym-hhfc/resources/trajectory/humanoid/run_50Hz_4p0_ori.dat"
    output_file = "/home/harmony/legged_gym-hhfc/resources/trajectory/humanoid/run_50Hz_4p0.dat"
    
    process_trajectory_file(input_file, output_file)
