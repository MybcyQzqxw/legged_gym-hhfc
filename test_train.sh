#!/bin/bash
# 快速测试训练脚本

echo "=== 测试 hhfc_rl 训练配置 ==="
echo ""

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate legged_gym

echo "Python 环境: $(which python)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# 运行1次迭代测试
echo "开始测试训练（1次迭代）..."
echo "----------------------------------------"
python legged_gym/scripts/train.py --task=hhfc_rl --headless --max_iterations=1

exit_code=$?
echo "----------------------------------------"
echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ 测试成功！配置正确。"
    echo ""
    echo "现在可以开始正式训练："
    echo "  python legged_gym/scripts/train.py --task=hhfc_rl --headless"
    echo ""
    echo "或者中等训练（5000次迭代）："
    echo "  python legged_gym/scripts/train.py --task=hhfc_rl --headless --max_iterations=5000"
else
    echo "❌ 测试失败，退出代码: $exit_code"
    echo ""
    echo "常见问题："
    echo "1. AttributeError - 配置缺少参数"
    echo "2. CUDA out of memory - 显存不足，使用 --num_envs=4096"
    echo "3. Import Error - 检查 isaacgym 是否正确安装"
    echo ""
    echo "请复制上面的完整错误信息寻求帮助。"
fi
