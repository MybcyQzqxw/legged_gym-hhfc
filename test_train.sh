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
python legged_gym/scripts/train.py --task=hhfc_rl --headless --max_iterations=1

exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ 测试成功！配置正确。"
    echo ""
    echo "现在可以开始正式训练："
    echo "  python legged_gym/scripts/train.py --task=hhfc_rl --headless"
else
    echo "❌ 测试失败，退出代码: $exit_code"
    echo "请检查上面的错误信息。"
fi
