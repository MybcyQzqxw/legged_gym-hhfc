# Hhfc_RL - 纯强化学习站立任务

## 📝 任务说明

`hhfc_rl` 是 Hhfc 人形机器人的**纯强化学习版本**站立任务。与 `hhfc` 任务的主要区别:

| 特性 | hhfc (模仿学习) | hhfc_rl (纯强化学习) |
|------|----------------|-------------------|
| **训练方法** | GAIL/AMP (模仿学习+RL) | 纯PPO (强化学习) |
| **训练器** | GAILRunner | OnPolicyRunner |
| **参考轨迹** | ✅ 需要 standup01.dat | ❌ 不需要 |
| **判别器** | ✅ 使用 | ❌ 不使用 |
| **主要驱动力** | imitation_state 奖励 (3.0) | stand_success (10.0) + base_height (3.0) |
| **is_amp** | True | False |
| **standup_imitation** | True | False |

## 🚀 使用方法

### 1. 训练

```bash
# 基础训练
python legged_gym/scripts/train.py --task=hhfc_rl

# 无头模式训练 (推荐, 更快)
python legged_gym/scripts/train.py --task=hhfc_rl --headless

# 自定义参数
python legged_gym/scripts/train.py --task=hhfc_rl --headless --num_envs=4096
```

### 2. 评估/播放

```bash
# 播放训练好的模型
python legged_gym/scripts/play.py --task=hhfc_rl

# 加载特定checkpoint
python legged_gym/scripts/play.py --task=hhfc_rl --checkpoint=5000
```

### 3. 继续训练

```bash
python legged_gym/scripts/train.py --task=hhfc_rl --resume --checkpoint=5000
```

## 📊 训练输出

训练日志和模型将保存在:
```
logs/hhfc_rl/
├── <date_time>_ppo/
│   ├── events.out.tfevents.*  (TensorBoard日志)
│   ├── model_0.pt
│   ├── model_200.pt
│   ├── model_400.pt
│   └── ...
```

使用TensorBoard查看训练曲线:
```bash
tensorboard --logdir=logs/hhfc_rl
```

## 🎯 核心奖励函数

纯RL版本的主要奖励包括:

### 主要驱动力
- **stand_success** (权重=10.0): 站立成功奖励
  - 高度 > 0.85m
  - |roll| < 0.25 rad
  - |pitch| < 0.25 rad
  
- **base_height** (权重=3.0): 基座高度奖励
  - 奖励 = exp(-10 * |height - 0.92|)

### 约束和平滑性
- **orientation** (-5.0): 姿态直立
- **ang_vel_xy** (-0.05): 减少roll/pitch晃动
- **action_smoothness** (-0.01): 动作平滑
- **torques** (-1e-5): 能耗约束

### 已禁用
- **tracking_lin_vel** (0.0): 不跟踪线速度
- **tracking_ang_vel** (0.0): 不跟踪角速度
- **imitation_state** (已删除): 不使用模仿学习

## 🔧 配置修改

如需修改训练参数,编辑 `legged_gym/envs/hhfc_rl/hhfc_rl_config.py`:

```python
class HhfcRlRobotCfgPPO:
    class runner:
        max_iterations = 20000      # 总迭代次数
        num_steps_per_env = 24      # 每次rollout步数
        
    class algorithm:
        learning_rate = 5.0e-4      # 学习率
        num_mini_batches = 4        # mini-batch数量
        
    class rewards:
        class scales:
            stand_success = 10.0    # 站立成功奖励权重
            base_height = 3.0       # 高度奖励权重
```

## 📈 预期训练效果

- **初期 (0-2000 iter)**: 机器人探索,可能频繁摔倒
- **中期 (2000-8000 iter)**: 学会部分站立动作,成功率逐渐提升
- **后期 (8000+ iter)**: 稳定站立,姿态控制良好

**注意**: 纯RL训练时间可能比模仿学习更长,需要更多探索。

## 🆚 与 hhfc 任务对比

### 何时使用 hhfc (模仿学习)?
- ✅ 有高质量的专家演示数据
- ✅ 需要快速收敛
- ✅ 模仿特定的动作风格

### 何时使用 hhfc_rl (纯强化学习)?
- ✅ 没有专家数据
- ✅ 希望策略自主发现最优解
- ✅ 追求更好的泛化能力
- ✅ 探索不同的解决方案

## 🐛 常见问题

### Q: 训练不收敛怎么办?
A: 尝试:
1. 增加 `stand_success` 和 `base_height` 的奖励权重
2. 减少其他惩罚项的权重
3. 调低学习率
4. 检查终止条件是否过于严格

### Q: 机器人摔倒太频繁?
A: 考虑:
1. 降低 `action_scale` (从0.25降到0.2)
2. 增加 `orientation` 奖励权重
3. 使用课程学习 (从简单初始状态开始)

### Q: 如何可视化训练过程?
A: 
```bash
# 训练时不使用 --headless
python legged_gym/scripts/train.py --task=hhfc_rl
```

## 📚 相关文件

- **配置文件**: `legged_gym/envs/hhfc_rl/hhfc_rl_config.py`
- **环境实现**: `legged_gym/envs/hhfc_rl/hhfc_rl.py`
- **任务注册**: `legged_gym/envs/__init__.py`
- **机器人模型**: `resources/robots/hhfc_sf/urdf/hhfc.urdf`

## 🎓 技术细节

### 观测空间 (450维)
- 10帧历史堆叠 × 45维单帧观测
- 单帧包含: 命令(3) + 关节位置(12) + 关节速度(12) + 上一动作(12) + 角速度(3) + 重力投影(3)

### 动作空间 (12维)
- 12个关节的目标角度增量
- PD控制器: τ = Kp(θ_target - θ) + Kd(0 - θ̇)

### 训练超参数
- 8192 并行环境
- 24 steps per rollout
- 学习率: 5e-4
- PPO clip: 0.2
- 折扣因子: 0.99

## 📞 支持

如有问题,请检查:
1. Isaac Gym 是否正确安装
2. PyTorch 版本是否兼容 (2.4.1+cu118)
3. CUDA 驱动是否正常

---

**创建时间**: 2025-11-06  
**版本**: v1.0  
**基于**: hhfc 模仿学习任务
