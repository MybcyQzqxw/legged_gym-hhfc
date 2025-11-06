from collections import deque

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hhfc_rl.hhfc_rl_config import HhfcRlRobotCfg
from legged_gym.utils.math import wrap_to_pi


class HhfcRlRobot(LeggedRobot):
    """Hhfc机器人纯强化学习环境类

    继承自LeggedRobot基类,实现Hhfc人形机器人通过纯强化学习学习站立动作。
    与HhfcRobot的区别: 移除了所有模仿学习相关的代码,完全依靠奖励函数引导学习。
    """

    cfg: HhfcRlRobotCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, actions):
        """环境步进函数 (增加控制延迟模拟)

        执行一个控制周期(包含decimation个仿真步),应用动作并更新环境状态。
        支持可选的控制延迟模拟以提高sim2real鲁棒性。

        Args:
            actions (Tensor): 策略输出的动作 (num_envs, num_actions)

        工作流程:
        1. 将动作裁剪到合理范围
        2. (可选) 将动作压入延迟队列,提取延迟后的动作
        3. 调用父类step()执行物理仿真
        4. 返回观测、特权观测、奖励、终止标志、额外信息

        Returns:
            obs_buf: 策略观测 (num_envs, num_observations)
            privileged_obs_buf: 特权观测 (num_envs, num_privileged_obs)
            rewards: 奖励 (num_envs,)
            dones: 终止标志 (num_envs,)
            infos: 额外信息字典
        """
        # 裁剪动作到合理范围
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # ========== 控制延迟模拟 ==========
        if self.cfg.domain_rand.randomize_ctrl_delay:
            # 将当前动作压入队列末尾
            self.action_queue = torch.cat(
                [self.action_queue[:, 1:], self.actions[:, None, :]], dim=1
            )
            # 从队列中提取延迟后的动作
            # action_delay 是每个环境的随机延迟步数 (0~max_delay)
            delayed_actions = torch.zeros_like(self.actions)
            for env_id in range(self.num_envs):
                delay = self.action_delay[env_id]
                # 负索引: -1是最新, -delay-1是延迟delay步后的动作
                delayed_actions[env_id] = self.action_queue[env_id, -delay - 1]
            self.actions = delayed_actions

        # 执行 decimation 次仿真步 (父类会处理)
        return super().step(self.actions)

    def compute_observations(self):
        """构造策略观测和特权观测

        策略观测 (num_observations = 450):
            由10帧历史堆叠而成,每帧45维:
            - commands(3): 线速度x, 线速度y, 角速度yaw
            - dof_pos(12): 当前关节位置
            - dof_vel(12): 当前关节速度
            - actions(12): 上一次的动作
            - base_ang_vel(3): 基座角速度 (body frame)
            - gravity(3): 重力投影 (body frame)

        特权观测 (num_privileged_obs = 243):
            由3帧历史堆叠而成,每帧81维:
            - 包含策略观测的所有信息(45维)
            - 额外特权信息(36维):
                * base_lin_vel(3): 基座线速度
                * friction_coeffs(1): 摩擦系数
                * body_mass(1): 机身质量
                * body_com(3): 质心位置
                * external_forces(6, 3): 外力 (只取前6个刚体)
                * contact_states(2): 双脚接触状态

        Notes:
            - 特权观测只在训练时使用(给critic),部署时不可用
            - 历史堆叠提供时序信息,帮助策略感知速度和加速度
        """
        # ========== 构造单帧观测(45维) ==========
        obs = torch.cat(
            [
                self.commands[:, :3] * self.commands_scale,  # [0:3] 速度命令
                self.dof_pos * self.obs_scales.dof_pos,  # [3:15] 关节位置
                self.dof_vel * self.obs_scales.dof_vel,  # [15:27] 关节速度
                self.actions,  # [27:39] 上一动作
                self.base_ang_vel * self.obs_scales.ang_vel,  # [39:42] 基座角速度
                self.projected_gravity,  # [42:45] 重力投影
            ],
            dim=-1,
        )

        # 添加观测噪声 (模拟传感器误差)
        if self.add_noise:
            obs += (
                2 * torch.rand_like(obs) - 1
            ) * self.noise_scale_vec  # U(-noise, +noise)

        # ========== 更新观测历史队列 ==========
        self.obs_history.append(obs)
        # 将历史帧拼接成完整观测 (10帧 × 45维 = 450维)
        self.obs_buf = torch.cat([tensor for tensor in self.obs_history], dim=-1)

        # ========== 构造特权观测(81维) ==========
        privileged_obs = torch.cat(
            [
                obs,  # [0:45] 包含全部策略观测
                self.base_lin_vel * self.obs_scales.lin_vel,  # [45:48] 基座线速度
                self.friction_coeffs[:, None],  # [48] 摩擦系数
                self.body_mass[:, None] / 20.0,  # [49] 归一化质量
                self.body_com,  # [50:53] 质心位置
                self.external_forces[:, :6, :].flatten(
                    start_dim=1
                ),  # [53:71] 前6个刚体的外力
                self.contact_states,  # [71:73] 双脚接触状态 (二值)
            ],
            dim=-1,
        )

        # 更新特权观测历史队列
        self.critic_history.append(privileged_obs)
        # 拼接成完整特权观测 (3帧 × 81维 = 243维)
        self.privileged_obs_buf = torch.cat(
            [tensor for tensor in self.critic_history], dim=-1
        )

    def post_physics_step(self):
        """每次物理步后的处理函数 (主循环)

        该函数在每个decimation周期后被调用,负责更新环境状态、计算奖励、处理重置等。

        执行流程 (9个步骤):
        1. 刷新仿真器状态张量 (关节、基座、刚体等)
        2. 更新episode计数器
        3. 计算观测相关量 (基座速度、重力投影、脚部状态等)
        4. 执行回调函数 (命令重采样、地形测量、外力扰动等)
        5. 检查终止条件 (摔倒、超时等)
        6. 计算奖励 (调用所有奖励函数)
        7. 重置终止的环境
        8. 计算观测 (策略观测和特权观测)
        9. 更新观测历史和临时变量

        Notes:
            - 终止条件包括: 摔倒(姿态角度过大)、膝盖碰地、超时
            - 重置时会清空历史队列,重新初始化状态
        """
        # ========== 步骤1: 刷新仿真器状态 ==========
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ========== 步骤2: 更新计数器 ==========
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # ========== 步骤3: 计算观测相关量 ==========
        # 提取关节位置和速度
        self.dof_pos[:] = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel[:] = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 提取基座位置、姿态、速度
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )  # 线速度: 世界系 -> body系
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )  # 角速度: 世界系 -> body系

        # 计算重力在body坐标系的投影 (用于姿态感知)
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        # 刷新脚部状态 (位置和速度)
        self._refresh_rigid_body_states()

        # ========== 步骤4: 执行通用回调 (命令重采样、地形高度测量、外力扰动等) ==========
        self._post_physics_step_callback()

        # ========== 步骤5: 计算终止条件和奖励 ==========
        self.check_termination()  # 检查哪些环境应该终止 (摔倒、超时等)
        self.compute_reward()  # 计算所有奖励项

        # ========== 步骤6: 重置终止的环境 ==========
        env_ids = self.reset_buf.nonzero(
            as_tuple=False
        ).flatten()  # 找出需要重置的环境ID
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # ========== 步骤7: 计算观测 (策略观测 + 特权观测) ==========
        self.compute_observations()

        # ========== 步骤8: 更新临时变量 (用于下一步的计算) ==========
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # ========== 步骤9: 可视化调试信息 (如果启用) ==========
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _reset_dofs(self, env_ids):
        """重置指定环境的关节状态 (随机初始化)

        为每个关节设置不同的随机初始角度范围,模拟从躺姿开始学习站立的场景。

        Args:
            env_ids (Tensor): 需要重置的环境索引列表

        关节初始化范围:
        - Lleg_hip_p (左髋pitch): default_dof_pos ± 0.5 rad
        - Lleg_knee (左膝): default_dof_pos + [-0.6, 0.1] rad (倾向弯曲)
        - Lleg_ankle_p (左踝pitch): default_dof_pos + [-0.1, 0.3] rad
        - 其他关节: default_dof_pos ± 0.1 rad
        - 右腿关节采用与左腿对称的随机范围

        Notes:
            - 关节速度初始化为0 (静止开始)
            - 随机初始化增加训练数据多样性,提高策略鲁棒性
        """
        # 从默认关节角度开始
        self.dof_pos[env_ids] = self.default_dof_pos

        # 为不同关节设置不同的随机范围
        # 左髋pitch: ±0.5 rad扰动
        self.dof_pos[env_ids, 0] += torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 左髋roll: ±0.1 rad扰动
        self.dof_pos[env_ids, 1] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 左髋yaw: ±0.1 rad扰动
        self.dof_pos[env_ids, 2] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 左膝: 偏向弯曲 [-0.6, 0.1] rad
        self.dof_pos[env_ids, 3] += torch_rand_float(
            -0.6, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 左踝pitch: 偏向背屈 [-0.1, 0.3] rad
        self.dof_pos[env_ids, 4] += torch_rand_float(
            -0.1, 0.3, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 左踝roll: ±0.1 rad扰动
        self.dof_pos[env_ids, 5] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # 右腿关节: 采用与左腿对称的随机范围
        # 右髋pitch
        self.dof_pos[env_ids, 6] += torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 右髋roll
        self.dof_pos[env_ids, 7] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 右髋yaw
        self.dof_pos[env_ids, 8] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 右膝
        self.dof_pos[env_ids, 9] += torch_rand_float(
            -0.6, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 右踝pitch
        self.dof_pos[env_ids, 10] += torch_rand_float(
            -0.1, 0.3, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # 右踝roll
        self.dof_pos[env_ids, 11] += torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # 关节速度初始化为0 (静止开始)
        self.dof_vel[env_ids] = 0.0

        # 计算实际环境ID (考虑多环境偏移)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 应用到仿真器
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """重置指定环境的基座状态 (躺姿初始化)

        设置机器人从躺姿开始,pitch角度约为-90度(面朝下躺着),
        模拟站立任务的起始状态。

        Args:
            env_ids (Tensor): 需要重置的环境索引列表

        初始化逻辑:
        1. 基座位置: init_state.pos + 小随机偏移
        2. 基座姿态: pitch ≈ -π/2 (躺姿) ± 0.05 rad随机扰动
        3. 基座速度: 全部设为0 (静止开始)

        Notes:
            - 躺姿开始使任务更具挑战性,需要策略学会完整的站立过程
            - 随机扰动增加初始状态多样性
        """
        # 从配置的初始状态开始
        self.root_states[env_ids] = self.base_init_state
        # 加上环境原点偏移
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # 速度初始化为0
        self.root_states[env_ids, 7:13] = 0

        # 设置躺姿: pitch ≈ -π/2 (面朝下躺着)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),  # roll = 0
            torch.tensor(-torch.pi / 2, device=self.device),  # pitch = -π/2
            torch.zeros(1, device=self.device),  # yaw = 0
        )

        # 添加姿态随机扰动 ±0.05 rad
        self.root_states[env_ids, 3:7] += torch_rand_float(
            -0.05, 0.05, (len(env_ids), 4), device=self.device
        )

        # 位置添加小随机偏移
        self.root_states[env_ids, :2] += torch_rand_float(
            -0.05, 0.05, (len(env_ids), 2), device=self.device
        )
        self.root_states[env_ids, 2] += torch_rand_float(
            -0.01, 0.01, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # 应用到仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        """重置指定环境的完整状态

        调用父类reset后,额外处理:
        1. 修正重力投影计算 (修复父类的bug)
        2. 清空观测历史队列 (避免不同episode的数据混合)
        3. 重置控制延迟队列

        Args:
            env_ids (Tensor): 需要重置的环境索引列表
        """
        # 调用父类重置 (关节、基座、外力等)
        super().reset_idx(env_ids)

        # 修复重力投影bug: 父类在reset后没有立即更新projected_gravity
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )

        # 清空观测历史队列 (避免不同episode的状态混合)
        for i in range(len(self.obs_history)):
            self.obs_history[i][env_ids] = 0
        for i in range(len(self.critic_history)):
            self.critic_history[i][env_ids] = 0

        # 重置控制延迟队列
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] = 0
            # 重新采样延迟步数
            self.action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
            )

    def _post_physics_step_callback(self):
        """物理步后的回调函数

        处理周期性任务:
        1. 命令重采样 (每resampling_time秒)
        2. 目标朝向控制 (如果启用heading_command)
        3. 地形高度测量 (如果启用measure_heights)
        4. 外力扰动 (每push_interval_s秒)

        Notes:
            - 重采样时机由episode_length_buf和resampling_time控制
            - 外力扰动用于测试策略的抗干扰能力
        """
        # 计算哪些环境需要重采样命令
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

        # 目标朝向控制 (如果启用)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        # 地形高度测量 (如果启用)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 外力扰动 (如果启用)
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _push_robots(self):
        """随机推动机器人

        通过直接设置基座速度模拟外力冲击,用于测试策略的鲁棒性。
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 最大推动速度
        # 在xy平面生成随机速度 [-max_vel, max_vel]
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        # 直接设置基座xy平面速度(模拟冲击效果)
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        # 应用到仿真器
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _refresh_rigid_body_states(self):
        """刷新刚体状态(主要是脚部状态)

        从仿真器获取的刚体状态中提取脚部的位置和速度,
        用于计算脚部相关的奖励(如着地速度、离地高度等)。
        """
        # 提取脚部速度: rigid_body_states[:, feet_indices, 7:10] = 线速度(x,y,z)
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        # 提取脚部位置: rigid_body_states[:, feet_indices, 0:3] = 位置(x,y,z)
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]

    def _get_noise_scale_vec(self, cfg):
        """构造观测噪声比例向量

        为每个观测维度分配噪声比例,用于训练时的噪声注入。
        通过在观测中添加噪声,提高策略对传感器误差的鲁棒性。
        [注意]: 修改观测结构时必须相应调整此函数

        Args:
            cfg (Dict): 环境配置对象,包含noise字段定义的噪声比例

        Returns:
            torch.Tensor: 形状为(num_single_obs,)的噪声比例向量,用于乘以[-1,1]均匀分布

        Notes:
            - num_single_obs = 45 (3命令 + 12关节位置 + 12关节速度 + 12上一动作 + 3角速度 + 3重力投影)
            - 已移除clock和phase输入
            - heights未使用(当前配置measure_heights=False),若启用需扩展num_single_obs
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device
        )  # 初始化噪声向量(45维)
        self.add_noise = self.cfg.noise.add_noise  # 是否添加噪声的开关
        noise_scales = self.cfg.noise.noise_scales  # 噪声比例配置
        noise_level = self.cfg.noise.noise_level  # 噪声强度系数
        # 为每个观测维度设置噪声比例:
        noise_vec[0:3] = 0.0  # [0:3] 命令 (不添加噪声,保持控制精度)
        # [3:15] 关节位置噪声
        noise_vec[3:15] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # [15:27] 关节速度噪声
        noise_vec[15:27] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[27:39] = 0.0  # [27:39] 上一动作 (不添加噪声,保持控制连续性)
        # [39:42] 角速度噪声
        noise_vec[39:42] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # [42:45] 重力投影噪声
        noise_vec[42:45] = noise_scales.gravity * noise_level
        return noise_vec

    def _init_buffers(self):
        """初始化仿真状态和处理量的张量缓冲区

        创建并初始化所有用于存储仿真状态、观测历史、动作队列等的PyTorch张量。
        继承父类初始化后,额外添加脚部状态、观测历史、动作延迟等特定缓冲区。

        Notes:
            - rigid_body_states: 刚体状态张量 (num_envs, num_bodies, 13) - 位置(3)+姿态(4)+线速度(3)+角速度(3)
            - obs_history: 观测历史队列,用于frame stacking (10帧 × 45维)
            - critic_history: 特权观测历史队列 (3帧 × 81维)
            - action_queue: 动作延迟队列 (最大延迟步数+1 × 12维动作)
            - action_delay: 每个环境的实际延迟步数 (随机初始化)
        """
        super()._init_buffers()  # 调用父类初始化(创建基础缓冲区)
        # ===== 获取刚体状态张量 (用于周期性奖励框架) =====
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 包装为PyTorch张量并reshape为(num_envs, num_bodies, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )
        # 提取脚部状态: 速度[7:10]和位置[0:3]
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]

        # ===== 初始化观测历史队列 (用于frame stacking) =====
        self.obs_history = deque(
            maxlen=self.cfg.env.frame_stack
        )  # actor的历史队列(10帧)
        self.critic_history = deque(
            maxlen=self.cfg.env.c_frame_stack
        )  # critic的历史队列(3帧)
        # 预填充actor历史队列(全零初始化)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,  # 单帧观测维度=45
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # 预填充critic历史队列(全零初始化)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,  # 单帧特权观测维度=81
                    dtype=torch.float,
                    device=self.device,
                )
            )

        # ===== 初始化控制延迟相关缓冲区 (如果启用) =====
        if self.cfg.domain_rand.randomize_ctrl_delay:
            # 动作队列: 存储历史动作以模拟延迟
            self.action_queue = torch.zeros(
                self.num_envs,
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,  # 最大延迟+1
                self.num_actions,  # 动作维度=12
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # 每个环境的随机延迟步数 [min, max]
            self.action_delay = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],  # 最小延迟
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,  # 最大延迟+1
                (self.num_envs,),
                device=self.device,
                requires_grad=False,
            )

    def _create_envs(self):
        """创建并配置仿真环境

        调用父类创建基础环境后,额外查找并存储特定刚体索引(脚部、膝盖)。
        这些索引用于后续的状态提取和奖励计算。

        Notes:
            - 继承LeggedRobot._create_envs()完成环境、地形、actor的创建
            - 区分左右脚索引 (foot_index_left, foot_index_right)
            - 查找膝盖刚体索引 (knee_indices) 用于碰撞检测和奖励计算
        """
        super()._create_envs()  # 调用父类创建环境、地形、加载URDF等

        # ===== 周期性奖励框架: 区分左右脚 =====
        for i in range(len(self.feet_names)):
            if "Lleg" in self.feet_names[i]:  # 左脚包含"Lleg"标识
                self.foot_index_left = self.feet_indices[i]
            elif "Rleg" in self.feet_names[i]:  # 右脚包含"Rleg"标识
                self.foot_index_right = self.feet_indices[i]

        # ===== 查找膝盖刚体索引 (用于碰撞检测) =====
        self.knee_indices = torch.zeros(
            len(self.knee_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        # 遍历膝盖名称,从第一个环境的第一个actor中查找刚体句柄
        for i in range(len(self.knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.knee_names[i]
            )

    # ================================================ 奖励函数 ================================================== #
    # 📝 说明: 这里是所有奖励函数的定义位置
    # 要修改/添加奖励函数, 请按照以下步骤:
    # 1. 在这个区域定义新的 _reward_xxx() 函数
    # 2. 在配置文件 hhfc_rl_config.py 的 rewards.scales 中添加对应的权重
    # 3. 权重为正数表示奖励(鼓励该行为), 为负数表示惩罚(抑制该行为)
    # 4. 父类 LeggedRobot 会自动调用所有 _reward_xxx() 函数并加权求和

    # ========== 核心任务奖励 (纯RL版本) ==========
    def _reward_base_height(self):
        """奖励基座高度接近目标 (纯RL核心奖励)

        使用指数函数奖励基座高度接近目标高度0.92m。
        这是纯RL版本的主要驱动力之一,引导机器人站起来。

        Returns:
            torch.Tensor: shape=[num_envs,], 高度越接近目标奖励越高

        配置权重: rewards.scales.base_height = 3.0
        """
        target_height = self.cfg.rewards.base_height_target  # 0.92m
        height_error = torch.abs(self.root_states[:, 2] - target_height)
        return torch.exp(-10.0 * height_error)

    def _reward_stand_success(self):
        """成功站立的奖励 (纯RL核心奖励)

        当机器人满足以下条件时给予奖励:
        1. 基座高度 > 0.85m
        2. roll角度绝对值 < 0.25 rad (约14度)
        3. pitch角度绝对值 < 0.25 rad

        Returns:
            torch.Tensor: shape=[num_envs,], 满足条件为1.0, 否则为0.0

        配置权重: rewards.scales.stand_success = 10.0 (主要驱动力!)
        """

        # 使用四元数计算欧拉角
        def quat_to_euler_xyz(quat):
            x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch_sin = torch.clamp(2.0 * (w * y - z * x), -0.999999, 0.999999)
            pitch = torch.asin(pitch_sin)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return roll, pitch, yaw

        roll, pitch, _ = quat_to_euler_xyz(self.base_quat)
        height = self.root_states[:, 2]

        # 判定条件
        height_ok = height > 0.85  # 高度足够
        roll_ok = torch.abs(roll) < 0.25  # roll角度足够小
        pitch_ok = torch.abs(pitch) < 0.25  # pitch角度足够小

        # 同时满足所有条件才给奖励
        success = height_ok & roll_ok & pitch_ok
        return success.float()

    # ========== 脚部相关奖励 ==========
    def _reward_foot_landing_vel(self):
        """惩罚脚部着地时的垂直速度

        在脚即将着地时检测其垂直速度，速度越大惩罚越大。
        目的是鼓励机器人轻柔着地，避免冲击过大。

        Returns:
            torch.Tensor: shape=[num_envs,], 着地速度的平方和（负奖励）

        配置权重: rewards.scales.foot_landing_vel = -0.2
        """
        z_vels = self.foot_vel[:, :, 2]  # 脚部在z方向的速度
        contacts = (
            self.contact_forces[:, self.feet_indices, 2] > 0.1
        )  # 判断是否已接触地面
        # 判断"即将着地"状态：脚高度低于阈值 且 未接触 且 向下运动
        about_to_land = (
            (self.foot_pos[:, :, 2] < self.cfg.rewards.about_landing_threshold)
            & (~contacts)
            & (z_vels < 0.0)
        )
        # 提取即将着地时的速度，其他时刻为0
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)  # 速度平方和
        return reward

    def _reward_feet_distance(self):
        """奖励合理的双脚间距

        鼓励双脚保持在合理范围内：既不能太近（避免碰撞），也不能太远（保持稳定）。
        使用双指数函数，在min_dist和max_dist附近给予高奖励。

        Returns:
            torch.Tensor: shape=[num_envs,], 范围 [0, 1]，最优间距时接近1

        配置权重: rewards.scales.feet_distance = 0.2
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]  # 双脚xy平面位置
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)  # 双脚距离
        fd = self.cfg.rewards.min_dist  # 最小允许距离（0.25m）
        max_df = self.cfg.rewards.max_dist  # 最大允许距离（0.6m）
        # 计算偏离合理范围的程度
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)  # 低于最小距离的偏差
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)  # 超过最大距离的偏差
        # 两个指数衰减函数的平均值：在合理范围内奖励高
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_knee_distance(self):
        """奖励合理的双膝间距

        类似双脚间距奖励，鼓励双膝保持适当距离。
        最大距离设为双脚最大距离的一半，因为膝关节活动范围较小。

        Returns:
            torch.Tensor: shape=[num_envs,], 范围 [0, 1]，最优间距时接近1

        配置权重: rewards.scales.knee_distance = 0.2
        """
        knee_pos = self.rigid_body_states[:, self.knee_indices, :2]  # 双膝xy平面位置
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)  # 双膝距离
        fd = self.cfg.rewards.min_dist  # 最小允许距离
        max_df = self.cfg.rewards.max_dist / 2  # 最大距离为双脚的一半
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_foot_clearance(self):
        """奖励摆动腿的离地高度

        在脚部快速移动（摆动相）时，鼓励脚离地一定高度，避免拖地。
        使用正向表述（指数形式），偏差越小奖励越高。

        Returns:
            torch.Tensor: shape=[num_envs,], 范围 (0, 1]，接近目标高度时接近1

        配置权重: rewards.scales.foot_clearance (当前未启用，默认0.0)
        """
        foot_vel_xy_norm = torch.norm(
            self.foot_vel[:, :, [0, 1]], dim=-1
        )  # 脚在xy平面的速度
        # 计算脚高度与目标的偏差，乘以移动速度（摆动时才计入）
        reward = torch.sum(
            foot_vel_xy_norm
            * torch.square(
                self.foot_pos[:, :, 2]  # 当前脚高度
                - self.cfg.rewards.foot_clearance_target  # 目标离地高度（0.08m）
                - self.cfg.rewards.foot_height_offset  # 基准偏移量（0.068m）
            ),
            dim=-1,
        )
        # 正向表述：偏差越小，奖励越接近1
        return torch.exp(-reward / 0.01)

    # ========== 平滑性和限制奖励 ==========
    def _reward_action_smoothness(self):
        """惩罚动作的不平滑（抖动）

        使用二阶差分（加速度）衡量动作平滑度。
        二阶差分 = a[t] - 2*a[t-1] + a[t-2]，值越大说明动作变化越剧烈。

        Returns:
            torch.Tensor: shape=[num_envs,], 动作加速度的平方和（负奖励）

        配置权重: rewards.scales.action_smoothness = -0.01
        """
        # 计算动作的二阶差分（离散加速度）
        action_smoothness_cost = torch.sum(
            torch.square(self.actions - 2 * self.last_actions + self.llast_actions),
            dim=-1,
        )
        return action_smoothness_cost

    def _reward_dof_pos_limits(self):
        """惩罚关节角度接近极限位置

        检测关节是否过于接近其物理限位，越接近惩罚越大。
        排除膝关节（索引3和9），因为膝关节通常需要大幅弯曲。

        Returns:
            torch.Tensor: shape=[num_envs,], 超出安全范围的总量（负奖励）

        配置权重: rewards.scales.dof_pos_limits = -2.0
        """
        dof_indices_excluding_knee = [
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            10,
            11,
        ]  # 排除膝关节（索引3和9）
        # 计算超出下限的程度
        out_of_limits = -(
            self.dof_pos[:, dof_indices_excluding_knee]
            - self.dof_pos_limits[dof_indices_excluding_knee, 0]
        ).clip(
            max=0.0
        )  # 低于下限时为正值
        # 累加超出上限的程度
        out_of_limits += (
            self.dof_pos[:, dof_indices_excluding_knee]
            - self.dof_pos_limits[dof_indices_excluding_knee, 1]
        ).clip(
            min=0.0
        )  # 高于上限时为正值
        return torch.sum(out_of_limits, dim=1)  # 所有关节的总超限量
