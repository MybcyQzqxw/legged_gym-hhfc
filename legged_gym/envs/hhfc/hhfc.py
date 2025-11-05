from collections import deque

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hhfc.hhfc_config import HhfcRobotCfg
from legged_gym.utils.math import wrap_to_pi


def quat_to_euler_xyz(quat: torch.Tensor):
    """Convert quaternion (x,y,z,w) to Euler angles (roll, pitch, yaw).
    Args:
        quat: (...,4) tensor
    Returns:
        roll, pitch, yaw tensors of shape (...,)
    """
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]
    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    # pitch (y-axis)
    t2 = 2.0 * (w * y - z * x)
    t2_clamped = torch.clamp(t2, -0.999999, 0.999999)
    pitch = torch.asin(t2_clamped)
    # yaw (z-axis)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return roll, pitch, yaw


class HhfcRobot(LeggedRobot):

    def test_ref_data(self):
        """测试参考轨迹数据的可视化播放函数

        该函数用于在Isaac Gym仿真器中逐帧播放参考轨迹数据,
        将专家演示动作在仿真环境中重现,用于验证轨迹数据的正确性和完整性。

        功能流程:
        1. 根据 num_states 配置解析参考轨迹文件的列布局
        2. 提取基座姿态、角速度、关节位置、关节速度等状态
        3. 在无限循环中逐帧设置机器人状态并渲染
        4. 播放完成后自动循环重播

        支持的状态维度:
        - 31维: 完整状态 (四元数4 + 角速度3 + 关节位置12 + 关节速度12)
        - 27维: 排除四元数 (角速度3 + 关节位置12 + 关节速度12)
        - 24维: 仅关节状态 (关节位置12 + 关节速度12)
        - 20维: 排除髋关节frontal自由度 (10关节位置 + 10关节速度)
        - 16维: 排除髋关节frontal和transversal (8关节位置 + 8关节速度)

        注意: 该函数会进入无限循环,需手动终止程序退出
        """
        self.ref_step_counter_state = 0  # 初始化帧计数器

        # 根据配置的状态维度,解析参考轨迹文件的数据列布局
        if self.cfg.env.num_states == 31:  # 31维: 包含四元数和基座角速度
            # 轨迹布局: [quat(4) | ang_vel(3) | dof_pos(12) | dof_vel(12)]
            ref_base_orientation = torch.tensor(
                self.ref_df_data[:, 0:4]
            ).cuda()  # 基座四元数(x,y,z,w)
            ref_base_ang_vel = torch.tensor(
                self.ref_df_data[:, 4:7]
            ).cuda()  # 基座角速度(roll,pitch,yaw rate)
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 7:19]).cuda()  # 12个关节角度
            ref_dof_vel = torch.tensor(
                self.ref_df_data[:, 19:31]
            ).cuda()  # 12个关节角速度

        elif self.cfg.env.num_states == 27:  # 27维: 包含角速度但不含四元数
            # 轨迹布局: [ang_vel(3) | dof_pos(12) | dof_vel(12)]
            # 四元数缺失,初始化为单位四元数(表示无旋转)
            ref_base_orientation = torch.zeros(
                (self.ref_df_data.shape[0], 4), dtype=torch.float
            ).cuda()
            ref_base_orientation[:, 3] = 1.0  # 设置w分量为1.0,表示单位四元数
            ref_base_ang_vel = torch.tensor(self.ref_df_data[:, 0:3]).cuda()
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 3:15]).cuda()
            ref_dof_vel = torch.tensor(self.ref_df_data[:, 15:27]).cuda()

        elif self.cfg.env.num_states == 24:  # 24维: 仅包含关节位置和速度
            # 轨迹布局: [dof_pos(12) | dof_vel(12)]
            # 基座姿态和角速度缺失,初始化为零(静止直立姿态)
            ref_base_orientation = torch.zeros(
                (self.ref_df_data.shape[0], 4), dtype=torch.float
            ).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros(
                (self.ref_df_data.shape[0], 3), dtype=torch.float
            ).cuda()
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 0:12]).cuda()
            ref_dof_vel = torch.tensor(self.ref_df_data[:, 12:24]).cuda()

        elif self.cfg.env.num_states == 20:  # 20维: 排除髋关节frontal自由度(索引1和7)
            # 轨迹布局: [dof_pos(10, excluding indices 1,7) | dof_vel(10, excluding indices 1,7)]
            # 排除的关节使用默认位置,速度设为0
            ref_base_orientation = torch.zeros(
                (self.ref_df_data.shape[0], 4), dtype=torch.float
            ).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros(
                (self.ref_df_data.shape[0], 3), dtype=torch.float
            ).cuda()
            # 初始化12维关节位置向量,排除的关节保持为0
            ref_dof_pos = torch.zeros(
                (self.ref_df_data.shape[0], 12), dtype=torch.float
            ).cuda()
            # 将轨迹中的10个关节数据填入对应索引(排除索引1和7)
            ref_dof_pos[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                self.ref_df_data[:, 0:10]
            ).cuda()
            ref_dof_vel = torch.zeros(
                (self.ref_df_data.shape[0], 12), dtype=torch.float
            ).cuda()
            ref_dof_vel[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                self.ref_df_data[:, 12:22]
            ).cuda()

        elif (
            self.cfg.env.num_states == 16
        ):  # 16维: 排除髋关节frontal(索引1,7)和transversal(索引2,8)
            # 轨迹布局: [dof_pos(8, excluding indices 1,2,7,8) | dof_vel(8, excluding indices 1,2,7,8)]
            # 排除更多髋关节自由度,仅保留8个主要关节
            ref_base_orientation = torch.zeros(
                (self.ref_df_data.shape[0], 4), dtype=torch.float
            ).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros(
                (self.ref_df_data.shape[0], 3), dtype=torch.float
            ).cuda()
            ref_dof_pos = torch.zeros(
                (self.ref_df_data.shape[0], 12), dtype=torch.float
            ).cuda()
            # 将轨迹中的8个关节数据填入对应索引(排除索引1,2,7,8)
            ref_dof_pos[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                self.ref_df_data[:, 0:8]
            ).cuda()
            ref_dof_vel = torch.zeros(
                (self.ref_df_data.shape[0], 12), dtype=torch.float
            ).cuda()
            ref_dof_vel[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                self.ref_df_data[:, 8:16]
            ).cuda()
        else:
            # 不支持的状态维度配置,抛出异常
            raise ValueError("Invalid number of states")

        # 无限循环播放轨迹数据,用于可视化验证
        while True:
            self.render()  # 更新仿真器画面显示

            # 步骤1: 设置当前帧的关节状态
            # 从轨迹数据中提取当前帧的关节位置和速度,并reshape为正确维度
            self.dof_pos[:] = ref_dof_pos[self.ref_step_counter_state, :].view(
                -1, self.num_dof
            )  # 关节位置: [num_envs, 12]
            self.dof_vel[:] = ref_dof_vel[self.ref_step_counter_state, :].view(
                -1, self.num_dof
            )  # 关节速度: [num_envs, 12]

            # 步骤2: 设置当前帧的基座状态
            self.root_states[:] = self.base_init_state  # 先重置为初始基座状态
            self.root_states[:, :3] += self.env_origins[
                :
            ]  # 加上环境原点偏移(多环境并行)
            # 设置基座姿态(四元数): root_states[:, 3:7] = [x, y, z, w]
            self.root_states[:, 3:7] = ref_base_orientation[
                self.ref_step_counter_state, :
            ].view(-1, 4)
            # 设置基座角速度: root_states[:, 10:13] = [roll_rate, pitch_rate, yaw_rate]
            self.root_states[:, 10:13] = ref_base_ang_vel[
                self.ref_step_counter_state, :
            ].view(-1, 3)

            # 步骤3: 将状态应用到仿真器
            # 获取所有环境的索引(用于批量更新)
            env_ids_int32 = torch.arange(self.num_envs, device=self.device).to(
                dtype=torch.int32
            )
            # 更新关节状态(位置和速度)到仿真器
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )
            # 更新基座状态(位置、姿态、速度)到仿真器
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

            # 步骤4: 执行一步仿真(用于测试状态是否合理)
            self.gym.simulate(self.sim)

            # 步骤5: 递增帧计数器,并在到达轨迹末尾时循环重播
            self.ref_step_counter_state += 1
            if self.ref_step_counter_state > self.ref_step_max:
                self.ref_step_counter_state = 0  # 重置计数器,从头播放

    def step(self, actions):
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[
                torch.arange(self.num_envs), self.action_delay
            ].clone()
        return super().step(actions)

    def compute_observations(self):
        """Computes observations"""
        obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
            ),
            dim=-1,
        )
        if self.cfg.domain_rand.randomize_ctrl_delay:
            ctrl_delay = (
                self.action_delay / self.cfg.domain_rand.ctrl_delay_step_range[1]
            ).view(
                -1, 1
            )  # normalize ctrl delay to [0, 1]
        # privileged obs add base height (z) -> expected 81 dims per config
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                self.rand_push_force[:, :2],  # 2
                self.env_frictions,  # 1
                self.base_mass / 30.0,  # 1
                self.com_displacements,  # 3
                self._kp_scale,  # 12
                self._kd_scale,  # 12
                self.joint_armature.unsqueeze(1),  # 1
                self.root_states[:, 2].unsqueeze(1),  # 1 base height
            ),
            dim=-1,
        )
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            obs_buf = torch.cat((obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        # obs_history
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
        )

    def compute_policy_state(self):
        """Computes policy_state for discriminator"""
        # 30-dim: base_euler(3) + base_ang_vel(3) + dof_pos(12) + dof_vel(12)
        roll, pitch, yaw = quat_to_euler_xyz(self.base_quat)
        base_euler = torch.stack((roll, pitch, yaw), dim=-1)
        self.state_buf = torch.cat(
            (base_euler, self.base_ang_vel, self.dof_pos, self.dof_vel), dim=-1
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # Periodic Reward Framework

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self._refresh_rigid_body_states()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_observations()

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # initialize dof_pos for each joint separately
        if self.cfg.init_state.init_joint_state_train:
            dof_pos = torch.zeros(
                (len(env_ids), self.num_dof), dtype=torch.float, device=self.device
            )  # 定义关节位置变量
            dof_pos[:, [0, 6]] = self.default_dof_pos[:, [0, 6]] + torch_rand_float(
                -0.5, 0.5, (len(env_ids), 2), device=self.device
            )  # hip_pitch
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [1, 7]] + torch_rand_float(
                -0.0, 0.0, (len(env_ids), 2), device=self.device
            )  # hip_roll
            dof_pos[:, [2, 8]] = self.default_dof_pos[:, [2, 8]] + torch_rand_float(
                -0.3, 0.3, (len(env_ids), 2), device=self.device
            )  # hip_yaw
            dof_pos[:, [3, 9]] = self.default_dof_pos[:, [3, 9]] + torch_rand_float(
                -0.6, 0.1, (len(env_ids), 2), device=self.device
            )  # knee_pitch
            dof_pos[:, [4, 10]] = self.default_dof_pos[:, [4, 10]] + torch_rand_float(
                -0.1, 0.3, (len(env_ids), 2), device=self.device
            )  # ankle_pitch
            dof_pos[:, [5, 11]] = self.default_dof_pos[:, [5, 11]] + torch_rand_float(
                -0.05, 0.05, (len(env_ids), 2), device=self.device
            )  # ankle_roll
            self.dof_pos[env_ids] = dof_pos[:]
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
            )

        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        # base orientation: lying start if standup imitation enabled
        if getattr(self.cfg.env, "standup_imitation", False):
            # Lie on back: pitch ~ -pi/2 (depending axis convention). Add small noise.
            pitch = -1.57 + torch_rand_float(
                -0.05, 0.05, (len(env_ids), 1), device=self.device
            ).view(-1)
            roll = torch_rand_float(
                -0.05, 0.05, (len(env_ids), 1), device=self.device
            ).view(-1)
            yaw = torch_rand_float(
                -3.14, 3.14, (len(env_ids), 1), device=self.device
            ).view(-1)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        else:
            base_orien_scale = self.cfg.init_state.init_base_angle_max
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
                torch_rand_float(
                    -base_orien_scale,
                    base_orien_scale,
                    (len(env_ids), 1),
                    device=self.device,
                ).view(-1),
                torch_rand_float(
                    -base_orien_scale,
                    base_orien_scale,
                    (len(env_ids), 1),
                    device=self.device,
                ).view(-1),
                torch_rand_float(
                    -base_orien_scale,
                    base_orien_scale,
                    (len(env_ids), 1),
                    device=self.device,
                ).view(-1),
            )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_dofs_gail(self, env_ids, frames):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: expert data frames to initialize motion with
        """
        if self.cfg.env.num_states == 31:  # contains quaternion, base angular velocity
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 7:19], dtype=torch.float
            ).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 19:31], dtype=torch.float
            ).cuda()
        elif self.cfg.env.num_states == 27:  # contains base angular velocity
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 3:15], dtype=torch.float
            ).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 15:27], dtype=torch.float
            ).cuda()
        elif self.cfg.env.num_states == 24:  # only contains joint angles and velocities
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 0:12], dtype=torch.float
            ).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 12:24], dtype=torch.float
            ).cuda()
        elif self.cfg.env.num_states == 20:  # exclude hip roll
            dof_pos = torch.zeros(
                (len(env_ids), self.num_dof), dtype=torch.float, device=self.device
            )
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [1, 7]]
            dof_pos[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                frames[:, 0:10], dtype=torch.float
            ).cuda()
            self.dof_pos[env_ids, :] = dof_pos[:]
            dof_vel = torch.zeros(
                (len(env_ids), self.num_dof), dtype=torch.float, device=self.device
            )
            dof_vel[:, [1, 7]] = 0.0
            dof_vel[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                frames[:, 10:20], dtype=torch.float
            ).cuda()
            self.dof_vel[env_ids, :] = dof_vel[:]
        elif self.cfg.env.num_states == 16:  # exclude hip roll and yaw
            dof_pos = torch.zeros(
                (len(env_ids), self.num_dof), dtype=torch.float, device=self.device
            )
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [1, 7]]
            dof_pos[:, [2, 8]] = self.default_dof_pos[:, [2, 8]] + torch_rand_float(
                -0.3, 0.3, (len(env_ids), 2), device=self.device
            )
            dof_pos[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                frames[:, 0:8], dtype=torch.float
            ).cuda()
            self.dof_pos[env_ids, :] = dof_pos[:]
            dof_vel = torch.zeros(
                (len(env_ids), self.num_dof), dtype=torch.float, device=self.device
            )
            dof_vel[:, [1, 7]] = 0.0
            dof_vel[:, [2, 8]] = 0.0
            dof_vel[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                frames[:, 8:16], dtype=torch.float
            ).cuda()
            self.dof_vel[env_ids, :] = dof_vel[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states_gail(self, env_ids, frames):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # ignore reference orientation
        self.root_states[env_ids, 3:7] = torch.zeros(
            (len(env_ids), 4), dtype=torch.float, device=self.device
        )
        self.root_states[env_ids, 6] = 1.0  # set w component of quaternion to 1.0
        # base velocities
        self.root_states[env_ids, 7:10] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 3), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.env.num_states == 31:  # contains quaternion, base angular velocity
            self.root_states[env_ids, 10:13] = torch.tensor(
                frames[:, 4:7], dtype=torch.float
            ).cuda()
        elif self.cfg.env.num_states == 27:  # contains base angular velocity
            self.root_states[env_ids, 10:13] = torch.tensor(
                frames[:, 0:3], dtype=torch.float
            ).cuda()
        else:  # only contains joint angles and velocities
            self.root_states[env_ids, 10:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 3), device=self.device
            )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )
        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        # randomize_ctrl_delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.0
            self.action_queue[env_ids] = 0.0
            self.action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )

    def _post_physics_step_callback(self):
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )
        # set small commands to zero
        self.commands[:, 2] *= (torch.abs(self.commands[:, 2]) > 0.1).float()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.root_states[:, 7:9] = self.rand_push_force[
            :, :2
        ]  # set random base velocity in xy plane
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _refresh_rigid_body_states(self):
        # Periodic Reward Framework
        # refresh the states of the rigid bodies
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device
        )
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0.0  # commands
        noise_vec[3:15] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[15:27] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[27:39] = 0.0  # previous actions
        noise_vec[39:42] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[42:45] = noise_scales.gravity * noise_level
        # removed clock & phase inputs; total length = 45
        # heights not used (measure_heights False in current config). If enabled, must extend num_single_obs accordingly.
        return noise_vec

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )  # Periodic Reward Framework
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # Periodic Reward Framework

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )  # Periodic Reward Framework
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs,
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                self.num_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.action_delay = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (self.num_envs,),
                device=self.device,
                requires_grad=False,
            )

    def _create_envs(self):
        super()._create_envs()
        # knee_dof_indices: [3,9]
        # Periodic Reward Framework. distinguish between 4 feet
        for i in range(len(self.feet_names)):
            if "Lleg" in self.feet_names[i]:
                self.foot_index_left = self.feet_indices[i]
            elif "Rleg" in self.feet_names[i]:
                self.foot_index_right = self.feet_indices[i]

        self.knee_indices = torch.zeros(
            len(self.knee_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(self.knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.knee_names[i]
            )

    # ================================================ Private Functions ================================================== #

    def _reward_foot_landing_vel(self):
        """惩罚脚部着地时的垂直速度

        在脚即将着地时检测其垂直速度，速度越大惩罚越大。
        目的是鼓励机器人轻柔着地，避免冲击过大。

        Returns:
            torch.Tensor: shape=[num_envs,]，着地速度的平方和（负奖励）
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
            torch.Tensor: shape=[num_envs,]，范围 [0, 1]，最优间距时接近1
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
            torch.Tensor: shape=[num_envs,]，范围 [0, 1]，最优间距时接近1
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
            torch.Tensor: shape=[num_envs,]，范围 (0, 1]，接近目标高度时接近1
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

    def _reward_action_smoothness(self):
        """惩罚动作的不平滑（抖动）

        使用二阶差分（加速度）衡量动作平滑度。
        二阶差分 = a[t] - 2*a[t-1] + a[t-2]，值越大说明动作变化越剧烈。

        Returns:
            torch.Tensor: shape=[num_envs,]，动作加速度的平方和（负奖励）
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
            torch.Tensor: shape=[num_envs,]，超出安全范围的总量（负奖励）
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

    # ---------------- Imitation Rewards -----------------
    def _build_current_state_30(self):
        """构建当前机器人的30维状态向量

        Returns:
            torch.Tensor: shape=[num_envs, 30]
                - 基座欧拉角 (3): roll, pitch, yaw
                - 基座角速度 (3): 世界坐标系下的角速度
                - 关节位置 (12): 12个关节的当前角度
                - 关节速度 (12): 12个关节的当前角速度
        """
        roll, pitch, yaw = quat_to_euler_xyz(self.base_quat)
        base_euler = torch.stack((roll, pitch, yaw), dim=-1)
        return torch.cat(
            (base_euler, self.base_ang_vel, self.dof_pos, self.dof_vel), dim=-1
        )

    def _get_ref_state(self):
        """从参考轨迹文件中获取当前时间步对应的专家状态

        使用懒加载机制，首次调用时从文件读取并转换为GPU张量。
        根据每个环境的episode进度，返回对应时刻的参考状态。

        Returns:
            torch.Tensor: shape=[num_envs, 30]，每个环境在其当前进度对应的专家状态
                如果episode超过轨迹长度，则返回最后一帧的状态
        """
        # 懒加载：仅在第一次调用时加载参考轨迹
        if not hasattr(self, "ref_state_tensor"):
            ref_np = self.ref_df_data  # 从文件读取的numpy数组
            # 验证轨迹文件列数是否满足要求
            if ref_np.shape[1] >= self.cfg.env.num_states:
                ref_np = ref_np[:, : self.cfg.env.num_states]  # 取前30列
            else:
                raise ValueError(
                    f"Reference trajectory columns {ref_np.shape[1]} < expected {self.cfg.env.num_states}"
                )
            # 转换为GPU张量以加速计算
            self.ref_state_tensor = torch.tensor(
                ref_np, dtype=torch.float, device=self.device
            )
            self.ref_state_len = self.ref_state_tensor.shape[0]  # 记录轨迹总帧数
        # 根据当前episode进度索引对应帧，防止越界
        idx = torch.clamp(self.episode_length_buf, max=self.ref_state_len - 1)
        return self.ref_state_tensor[idx]

    def _reward_imitation_state(self):
        """计算状态模仿奖励

        对比机器人当前状态与参考轨迹中对应时刻的专家状态，
        计算均方误差并转换为指数形式的奖励信号。

        Returns:
            torch.Tensor: shape=[num_envs,]，范围 (0, 1]
                状态越接近专家演示，奖励越高（接近1）
                状态偏差越大，奖励越低（接近0）
        """
        cur = self._build_current_state_30()  # 机器人当前30维状态
        ref = self._get_ref_state()  # 参考轨迹对应时刻的30维状态
        err = torch.sum((cur - ref) ** 2, dim=-1) / cur.shape[-1]  # 归一化的均方误差
        # 使用指数函数放大惩罚：误差越大，奖励衰减越快
        return torch.exp(-10.0 * err)

    def _reward_stand_success(self):
        """判断机器人是否成功站立

        通过检查基座高度和姿态角度，判断机器人是否达到直立站立状态。

        Returns:
            torch.Tensor: shape=[num_envs,]，值为0或1
                1.0: 成功站立（高度>0.85m 且 姿态接近竖直）
                0.0: 未站立
        """
        # 获取姿态角度
        roll, pitch, _ = quat_to_euler_xyz(self.base_quat)
        height = self.root_states[:, 2]  # 基座高度（z坐标）
        # 判断条件：高度足够高 且 roll和pitch角度接近0（±0.25弧度 ≈ ±14度）
        upright = (height > 0.85) & (torch.abs(roll) < 0.25) & (torch.abs(pitch) < 0.25)
        return upright.float()  # 转换为浮点数奖励
