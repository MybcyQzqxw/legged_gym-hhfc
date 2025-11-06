# Hhfc_RL 完整版本说明与奖励函数修改指南

## 📝 完整版本 vs 简易版本对比

### 之前简易版本的缺失内容

我之前创建的简易版本**缺失了以下重要内容**:

#### 1. ❌ 缺少的奖励函数
简易版只有 5 个奖励函数，**完整版应该有 8+ 个**:

| 奖励函数 | 简易版 | 完整版 | 用途 |
|---------|--------|--------|------|
| `_reward_base_height()` | ✅ | ✅ | **核心**: 基座高度奖励 |
| `_reward_stand_success()` | ✅ | ✅ | **核心**: 站立成功判定 |
| `_reward_foot_landing_vel()` | ✅ | ✅ | 着地速度惩罚 |
| `_reward_feet_distance()` | ✅ | ✅ | 双脚间距奖励 |
| `_reward_knee_distance()` | ✅ | ✅ | 双膝间距奖励 |
| `_reward_foot_clearance()` | ❌ | ✅ | 摆动腿离地高度 |
| `_reward_action_smoothness()` | ✅ | ✅ | 动作平滑度 |
| `_reward_dof_pos_limits()` | ❌ | ✅ | 关节限位惩罚 |

#### 2. ❌ 简化的函数实现
- `_reward_foot_landing_vel()`: 简易版逻辑不完整
- `_reward_stand_success()`: 简易版用了简化的角度计算

#### 3. ❌ 缺少详细的中文注释
- 简易版注释不够详细
- 完整版每个函数都有:
  - 详细的功能说明
  - 参数和返回值说明
  - **配置权重位置说明** ← 这个很重要!

---

## 🎯 奖励函数详解与修改指南

### 核心概念

奖励函数系统由**两部分**组成:

1. **函数定义**: `legged_gym/envs/hhfc_rl/hhfc_rl.py` 中的 `_reward_xxx()` 函数
2. **权重配置**: `legged_gym/envs/hhfc_rl/hhfc_rl_config.py` 中的 `rewards.scales.xxx`

**工作流程**:
```
训练循环
  ↓
compute_reward() [父类LeggedRobot中]
  ↓
自动查找所有 _reward_xxx() 函数
  ↓
调用每个函数计算奖励值
  ↓
乘以对应的 rewards.scales.xxx 权重
  ↓
加权求和得到总奖励
  ↓
传递给PPO算法
```

---

## 📍 奖励函数定义位置

### 文件: `legged_gym/envs/hhfc_rl/hhfc_rl.py`

**位置**: 文件末尾的 "奖励函数" 区域 (约 line 600+)

```python
# ================================================ 奖励函数 ================================================== #
# 📝 说明: 这里是所有奖励函数的定义位置
# 要修改/添加奖励函数, 请按照以下步骤...

def _reward_base_height(self):
    """奖励基座高度接近目标"""
    # 函数实现...
    
def _reward_stand_success(self):
    """成功站立的奖励"""
    # 函数实现...
    
# ... 更多奖励函数
```

### 命名规则

✅ **必须**以 `_reward_` 开头，例如:
- `_reward_base_height`
- `_reward_stand_success`
- `_reward_my_custom_reward` ← 你的自定义奖励

❌ **不能**使用其他前缀:
- `reward_xxx` ← 缺少下划线，不会被识别
- `_my_reward` ← 缺少 reward 关键字
- `calculate_reward` ← 完全错误的命名

---

## ⚙️ 奖励权重配置位置

### 文件: `legged_gym/envs/hhfc_rl/hhfc_rl_config.py`

**位置**: `HhfcRlRobotCfg.rewards.scales` 类 (约 line 140-180)

```python
class HhfcRlRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        # 奖励参数配置 (阈值、目标等)
        base_height_target = 0.92
        min_dist = 0.25
        max_dist = 0.6
        
        class scales(LeggedRobotCfg.rewards.scales):
            """各奖励项的权重系数"""
            
            # 核心任务奖励
            stand_success = 10.0      # ← 对应 _reward_stand_success()
            base_height = 3.0         # ← 对应 _reward_base_height()
            
            # 约束惩罚
            dof_pos_limits = -2.0     # ← 对应 _reward_dof_pos_limits()
            action_smoothness = -0.01 # ← 对应 _reward_action_smoothness()
            
            # ... 更多权重
```

### 权重规则

- **正数** (>0): 鼓励该行为，返回值越大奖励越多
  - 例如: `stand_success = 10.0` → 站起来给 +10 奖励
  
- **负数** (<0): 惩罚该行为，返回值越大惩罚越多
  - 例如: `action_smoothness = -0.01` → 动作抖动惩罚 -0.01
  
- **0**: 完全禁用该奖励
  - 例如: `tracking_lin_vel = 0.0` → 不跟踪速度

---

## 🔧 如何添加/修改奖励函数

### 场景 1: 修改现有奖励的权重

**最简单! 只需修改配置文件**

1. 打开 `legged_gym/envs/hhfc_rl/hhfc_rl_config.py`
2. 找到 `class scales` 区域
3. 修改对应的权重值

**示例**: 让站立奖励更强
```python
# 修改前
stand_success = 10.0

# 修改后 (增强2倍)
stand_success = 20.0
```

**示例**: 减少平滑度惩罚
```python
# 修改前
action_smoothness = -0.01

# 修改后 (减弱50%)
action_smoothness = -0.005
```

### 场景 2: 禁用某个奖励

**只需将权重设为 0**

```python
# 禁用脚部间距奖励
feet_distance = 0.0

# 禁用膝盖间距奖励
knee_distance = 0.0
```

### 场景 3: 添加新的奖励函数

**需要修改两个文件**

#### 步骤 1: 在 `hhfc_rl.py` 中定义函数

```python
# legged_gym/envs/hhfc_rl/hhfc_rl.py
# 在奖励函数区域添加:

def _reward_energy_efficiency(self):
    """奖励能量效率 (新增奖励示例)
    
    惩罚大力矩输出，鼓励节能。
    
    Returns:
        torch.Tensor: shape=[num_envs,], 力矩平方和
    
    配置权重: rewards.scales.energy_efficiency = -0.001
    """
    # 计算所有关节的力矩平方和
    torque_squared = torch.sum(torch.square(self.torques), dim=1)
    return torque_squared
```

#### 步骤 2: 在 `hhfc_rl_config.py` 中添加权重

```python
# legged_gym/envs/hhfc_rl/hhfc_rl_config.py
class scales(LeggedRobotCfg.rewards.scales):
    """各奖励项的权重系数"""
    
    # ... 现有的奖励权重 ...
    
    # 新增的能量效率惩罚
    energy_efficiency = -0.001  # ← 添加这一行
```

#### 步骤 3: 重新训练

```bash
python legged_gym/scripts/train.py --task=hhfc_rl --headless
```

系统会**自动识别**新的 `_reward_energy_efficiency()` 函数并使用!

---

## 📊 当前完整版本的所有奖励函数

### 核心任务奖励 (纯RL驱动力)

| 函数名 | 权重 | 说明 | 返回值范围 |
|--------|------|------|-----------|
| `_reward_stand_success` | **10.0** | 站立成功判定 (高度>0.85m且姿态直立) | 0 或 1 |
| `_reward_base_height` | **3.0** | 基座高度接近0.92m | (0, 1] |

### 脚部相关奖励

| 函数名 | 权重 | 说明 | 返回值范围 |
|--------|------|------|-----------|
| `_reward_foot_landing_vel` | **-0.2** | 惩罚着地时的垂直速度 | ≥0 |
| `_reward_feet_distance` | **0.2** | 双脚间距在[0.25, 0.6]m内 | [0, 1] |
| `_reward_knee_distance` | **0.2** | 双膝间距合理 | [0, 1] |
| `_reward_foot_clearance` | **0.0** | 摆动腿离地高度 (当前禁用) | (0, 1] |

### 平滑性和限制

| 函数名 | 权重 | 说明 | 返回值范围 |
|--------|------|------|-----------|
| `_reward_action_smoothness` | **-0.01** | 惩罚动作抖动 (二阶差分) | ≥0 |
| `_reward_dof_pos_limits` | **-2.0** | 惩罚关节接近限位 | ≥0 |

### 继承自父类的奖励 (自动启用)

这些奖励在 `LeggedRobot` 父类中定义，权重在配置文件中设置:

| 函数名 | 权重 | 说明 |
|--------|------|------|
| `_reward_termination` | **-200.0** | 摔倒或超时的巨大惩罚 |
| `_reward_dof_vel_limits` | **-1.0** | 关节速度超限 |
| `_reward_dof_acc` | **-2.5e-7** | 关节加速度惩罚 |
| `_reward_dof_vel` | **-5.0e-4** | 关节速度惩罚 |
| `_reward_action_rate` | **-0.01** | 动作变化率 |
| `_reward_torques` | **-1.0e-5** | 力矩惩罚 |
| `_reward_collision` | **-1.0** | 不期望的碰撞 |
| `_reward_lin_vel_z` | **-0.5** | 垂直速度惩罚 |
| `_reward_ang_vel_xy` | **-0.05** | roll/pitch角速度惩罚 |
| `_reward_orientation` | **-5.0** | 姿态偏离惩罚 |
| `_reward_tracking_lin_vel` | **0.0** | 线速度跟踪 (已禁用) |
| `_reward_tracking_ang_vel` | **0.0** | 角速度跟踪 (已禁用) |

---

## 🧪 奖励函数调优技巧

### 技巧 1: 从大到小调整

1. **先设置核心奖励** (主要驱动力)
   ```python
   stand_success = 10.0  # 很大的正奖励
   base_height = 3.0
   ```

2. **再添加约束惩罚** (防止坏行为)
   ```python
   termination = -200.0  # 很大的惩罚
   orientation = -5.0
   ```

3. **最后微调细节** (优化动作质量)
   ```python
   action_smoothness = -0.01  # 小的惩罚
   torques = -1.0e-5  # 很小的惩罚
   ```

### 技巧 2: 使用 TensorBoard 监控

训练时查看每个奖励项的数值:

```bash
tensorboard --logdir=logs/hhfc_rl
```

在 "Scalars" 标签下查看:
- `rewards/stand_success_mean`: 平均站立成功率
- `rewards/base_height_mean`: 平均高度奖励
- `rewards/total`: 总奖励

**调优准则**:
- 如果某个奖励的绝对值**远大于其他奖励**，说明权重可能过大
- 如果某个奖励始终接近 0，说明权重可能过小或函数设计有问题

### 技巧 3: 渐进式课程学习

从简单到困难:

**阶段 1** (0-5000 iter): 只学站立
```python
stand_success = 20.0    # 大奖励
action_smoothness = 0.0 # 暂时不管平滑度
```

**阶段 2** (5000-10000 iter): 优化姿态
```python
stand_success = 10.0    # 降低
orientation = -10.0     # 增强姿态要求
```

**阶段 3** (10000+ iter): 精细化
```python
action_smoothness = -0.01  # 启用平滑度
torques = -1.0e-5         # 启用能耗约束
```

---

## 🔍 调试奖励函数

### 查看单个奖励的数值

在 `hhfc_rl.py` 中临时添加打印:

```python
def _reward_stand_success(self):
    """成功站立的奖励"""
    # ... 原有代码 ...
    success = height_ok & roll_ok & pitch_ok
    
    # 临时调试: 每100步打印一次
    if self.common_step_counter % 100 == 0:
        print(f"Stand success rate: {success.float().mean().item():.2%}")
        print(f"Mean height: {height.mean().item():.3f}m")
    
    return success.float()
```

### 可视化奖励趋势

使用 TensorBoard 的 Scalars 功能:
- X轴: 训练迭代次数
- Y轴: 奖励数值
- 观察趋势: 应该逐渐上升(奖励)或下降(惩罚)

---

## ⚠️ 常见错误

### 错误 1: 函数名不符合规范

```python
# ❌ 错误
def reward_my_custom(self):  # 缺少前导下划线
    pass

# ✅ 正确
def _reward_my_custom(self):
    pass
```

### 错误 2: 忘记添加权重配置

```python
# 在 hhfc_rl.py 中定义了函数:
def _reward_my_custom(self):
    return torch.zeros(self.num_envs)

# ❌ 但在 hhfc_rl_config.py 中忘记添加:
# (缺少: my_custom = 1.0)

# 结果: 该奖励不会生效 (默认权重为0)
```

### 错误 3: 返回值形状错误

```python
# ❌ 错误: 返回标量
def _reward_wrong(self):
    return 1.0  # 只有一个数字

# ✅ 正确: 返回向量
def _reward_correct(self):
    return torch.ones(self.num_envs, device=self.device)  # [8192,]
```

---

## 📚 总结

### 奖励函数修改的两个文件

1. **函数定义**: `legged_gym/envs/hhfc_rl/hhfc_rl.py`
   - 位置: 文件末尾的 "奖励函数" 区域
   - 命名: 必须以 `_reward_` 开头
   - 返回: shape=[num_envs,] 的 Tensor

2. **权重配置**: `legged_gym/envs/hhfc_rl/hhfc_rl_config.py`
   - 位置: `HhfcRlRobotCfg.rewards.scales` 类
   - 命名: 与函数名对应 (去掉 `_reward_` 前缀)
   - 取值: 正数=奖励, 负数=惩罚, 0=禁用

### 快速参考

```python
# 1. 定义奖励函数 (hhfc_rl.py)
def _reward_xxx(self):
    # 计算奖励值
    reward = ...
    return reward  # shape=[num_envs,]

# 2. 设置权重 (hhfc_rl_config.py)
class scales:
    xxx = 1.0  # 对应 _reward_xxx()

# 3. 训练
python legged_gym/scripts/train.py --task=hhfc_rl --headless
```

---

**创建时间**: 2025-11-06  
**文档版本**: v2.0 (完整版)  
**对应代码**: hhfc_rl.py 完整版
