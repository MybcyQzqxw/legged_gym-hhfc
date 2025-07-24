# Legged Gym

*其他语言版本: [English](README.md), [中文](README.zh-CN.md)*

## 安装步骤

1. 创建Python 3.8虚拟环境：
   ```bash
   conda create --name legged_gym python=3.8 -y
   conda activate legged_gym
   ```

2. 安装支持CUDA-11.8的PyTorch：
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. 安装Isaac Gym：
   - 从[NVIDIA官网](https://developer.nvidia.com/isaac-gym)下载并安装Isaac Gym Preview 4
   - 安装Python包：
   ```bash
   cd isaacgym/python && pip install -e .
   ```
   - 测试示例：
   ```bash
   cd examples && python 1080_balls_of_solitude.py
   ```
   - 如遇问题，请查阅文档：`isaacgym/docs/index.html`

4. 安装本项目：
   - 安装本项目库：
   ```bash
   cd legged_gym-hhfc && pip install -e .
   ```
   - 安装RL库：
   ```bash
   cd rsl_rl && pip install -e .
   ```
   - 安装主项目：
   ```bash
   cd .. && pip install -e .
   ```
   - 安装额外依赖：
   ```bash
   pip install tensorboard pandas
   ```

## 代码结构

1. 每个环境由两个文件定义：
   - 环境文件（如`legged_robot.py`）
   - 配置文件（如`legged_robot_config.py`）
   
   配置文件包含两个类：环境参数类（`LeggedRobotCfg`）和训练参数类（`LeggedRobotCfgPPo`）

2. 环境和配置类都使用继承机制

3. 在`cfg`中指定的每个非零奖励值都会添加一个相应的函数，这些函数的返回值会被求和得到总奖励

4. 任务需要通过`task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`进行注册，这一步在`envs/__init__.py`中完成

## 使用方法

1. **训练模型**：
   ```bash
   cd legged_gym-hhfc/legged_gym/scripts && python train.py
   ```
   - 在CPU上运行：添加参数 `--sim_device=cpu --rl_device=cpu`
   - 无渲染运行：添加参数 `--headless`
   - **重要提示**：为提高性能，训练开始后按`v`停止渲染，可在需要查看进度时再启用
   - 训练好的策略保存在`issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`
   - 支持的命令行参数包括：
     - `--task TASK`：任务名称
     - `--resume`：从检查点恢复训练
     - `--experiment_name EXPERIMENT_NAME`：实验名称
     - `--run_name RUN_NAME`：运行名称
     - `--load_run LOAD_RUN`：当resume=True时要加载的运行名称
     - `--checkpoint CHECKPOINT`：保存的模型检查点编号
     - `--num_envs NUM_ENVS`：创建的环境数量
     - `--seed SEED`：随机种子
     - `--max_iterations MAX_ITERATIONS`：训练的最大迭代次数

2. **运行训练好的策略**：
   ```bash
   python issacgym_anymal/scripts/play.py --task=anymal_c_flat
   ```
   - 默认加载的是实验文件夹中最后一次运行的最后一个模型
   - 可以通过在训练配置中设置`load_run`和`checkpoint`来选择其他运行/模型迭代

3. **可视化参考数据**：
   ```bash
   cd legged_gym-hhfc/legged_gym/scripts && python test_humanoid_ref_data.py
   ```

## 添加新环境

基础环境`legged_robot`实现了粗糙地形的四足运动任务。对应的配置文件没有指定机器人资产（URDF/MJCF）和奖励比例。

1. 在`envs/`中添加新文件夹，包含继承自现有环境配置的`<your_env>_config.py`

2. 添加新机器人时：
   - 将相应资产添加到resources
   - 在`cfg`中设置资产路径、定义身体名称、默认关节位置和PD增益
   - 在`train_cfg`中设置`experiment_name`和`run_name`

3. 如有需要，在`<your_env>.py`中实现环境，继承现有环境，重写所需函数或添加奖励函数

4. 在`isaacgym_anymal/envs/__init__.py`中注册新环境

5. 根据需要修改/调整`cfg`和`cfg_train`中的参数。将奖励比例设为零可以移除该奖励

项目结构包含主要模块legged_gym和rsl_rl，前者包含环境定义，后者包含强化学习算法实现。