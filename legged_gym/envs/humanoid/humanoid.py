from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot


from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import os
import torch
import pandas as pd
from legged_gym.envs.humanoid.humanoid_config import HumanoidRobotCfg
from legged_gym.utils.math import wrap_to_pi
from scipy.stats import vonmises
from collections import deque


class HumanoidRobot(LeggedRobot):

    def test_ref_data(self):
        ''' 读取参考轨迹，和轨迹长度'ref_step_max' '''
        
        self.ref_step_counter_state = 0
        
        # determine reference data according to num_states
        if self.cfg.env.num_states == 31:  # contains quaternion, base angular velocity
            ref_base_orientation = torch.tensor(self.ref_df_data[:, 0:4]).cuda()
            ref_base_ang_vel = torch.tensor(self.ref_df_data[:, 4:7]).cuda()
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 7:19]).cuda()
            ref_dof_vel = torch.tensor(self.ref_df_data[:, 19:31]).cuda()
        elif self.cfg.env.num_states == 27:  # contains base angular velocity
            ref_base_orientation = torch.zeros((self.ref_df_data.shape[0], 4), 
                                               dtype=torch.float).cuda()
            ref_base_orientation[:, 3] = 1.0  # set w component to 1.0
            ref_base_ang_vel = torch.tensor(self.ref_df_data[:, 0:3]).cuda()
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 3:15]).cuda()
            ref_dof_vel = torch.tensor(self.ref_df_data[:, 15:27]).cuda()
        elif self.cfg.env.num_states == 24:  # only include dof_pos and dof_vel
            ref_base_orientation = torch.zeros((self.ref_df_data.shape[0], 4), 
                                               dtype=torch.float).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros((self.ref_df_data.shape[0], 3), 
                                               dtype=torch.float).cuda()
            ref_dof_pos = torch.tensor(self.ref_df_data[:, 0:12]).cuda()
            ref_dof_vel = torch.tensor(self.ref_df_data[:, 12:24]).cuda()
        elif self.cfg.env.num_states == 20: # exclude hip frontal
            ref_base_orientation = torch.zeros((self.ref_df_data.shape[0], 4),
                                                  dtype=torch.float).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros((self.ref_df_data.shape[0], 3),
                                                  dtype=torch.float).cuda()
            ref_dof_pos = torch.zeros((self.ref_df_data.shape[0], 12),
                                      dtype=torch.float).cuda()
            ref_dof_pos[:, [0,2,3,4,5,6,8,9,10,11]] = torch.tensor(
                self.ref_df_data[:, 0:10]).cuda()
            ref_dof_vel = torch.zeros((self.ref_df_data.shape[0], 12),
                                      dtype=torch.float).cuda()
            ref_dof_vel[:, [0,2,3,4,5,6,8,9,10,11]] = torch.tensor(
                self.ref_df_data[:, 12:22]).cuda()
        elif self.cfg.env.num_states == 16: # exclude hip frontal and hip transversal
            ref_base_orientation = torch.zeros((self.ref_df_data.shape[0], 4),
                                                  dtype=torch.float).cuda()
            ref_base_orientation[:, 3] = 1.0
            ref_base_ang_vel = torch.zeros((self.ref_df_data.shape[0], 3),
                                                  dtype=torch.float).cuda()
            ref_dof_pos = torch.zeros((self.ref_df_data.shape[0], 12),
                                      dtype=torch.float).cuda()
            ref_dof_pos[:, [0,3,4,5,6,9,10,11]] = torch.tensor(
                self.ref_df_data[:, 0:8]).cuda()
            ref_dof_vel = torch.zeros((self.ref_df_data.shape[0], 12),
                                        dtype=torch.float).cuda()
            ref_dof_vel[:, [0,3,4,5,6,9,10,11]] = torch.tensor(
                self.ref_df_data[:, 8:16]).cuda()
        else:
            raise ValueError("Invalid number of states")
        
        while True:
            self.render()
            self.dof_pos[:] = ref_dof_pos[self.ref_step_counter_state, :].view(-1, self.num_dof)
            self.dof_vel[:] = ref_dof_vel[self.ref_step_counter_state, :].view(-1, self.num_dof)
            self.root_states[:] = self.base_init_state
            self.root_states[:, :3] += self.env_origins[:]
            self.root_states[:, 3:7] = ref_base_orientation[self.ref_step_counter_state, :].view(-1, 4)
            self.root_states[:, 10:13] = ref_base_ang_vel[self.ref_step_counter_state, :].view(-1, 3)
            env_ids_int32 = torch.arange(self.num_envs, device=self.device).to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            # for testing state
            self.gym.simulate(self.sim)
            self.ref_step_counter_state += 1
            if self.ref_step_counter_state > self.ref_step_max:
                self.ref_step_counter_state = 0

    def step(self, actions):
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(
                self.num_envs), self.action_delay].clone()
        return super().step(actions)

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            (self.dof_pos - self.default_dof_pos) *
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,                         # 12
            self.actions,                                                   # 12
            self.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            self.projected_gravity,                                         # 3
        ), dim=-1)
        if self.cfg.domain_rand.randomize_ctrl_delay:
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).view(-1, 1)  # normalize ctrl delay to [0, 1]
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,                       # 3
            self.commands[:, :3] * self.commands_scale,   # 3
            (self.dof_pos - self.default_dof_pos) *
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,       # 12
            self.actions,                                 # 12
            self.rand_push_force[:, :2],                  # 2
            self.env_frictions,                           # 1
            self.base_mass / 30.0,                        # 1
            self.com_displacements,                       # 3
            # ctrl_delay,                                   # 1
            self._kp_scale,                               # 12
            self._kd_scale,                               # 12
            self.joint_armature.unsqueeze(1),             # 1
        ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            obs_buf = torch.cat((obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
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
        """ Computes policy_state for discriminator
        """
        self.state_buf = torch.cat((
                                    # self.base_quat,  # 4
                                #    self.base_ang_vel, # 3
                                   self.dof_pos,
                                   self.dof_vel,
                                    ), dim=-1)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(
            self.sim)  # Periodic Reward Framework

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
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
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # initialize dof_pos for each joint separately
        if self.cfg.init_state.init_joint_state_train:
            dof_pos = torch.zeros((len(env_ids), self.num_dof),
                                  dtype=torch.float, device=self.device)  # 定义关节位置变量
            dof_pos[:, [0, 6]] = self.default_dof_pos[:, [
                0, 6]] + torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)  # hip_pitch
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [
                1, 7]] + torch_rand_float(-0.0, 0.0, (len(env_ids), 2), device=self.device)  # hip_roll
            dof_pos[:, [2, 8]] = self.default_dof_pos[:, [
                2, 8]] + torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)  # hip_yaw
            dof_pos[:, [3, 9]] = self.default_dof_pos[:, [
                3, 9]] + torch_rand_float(-0.6, 0.1, (len(env_ids), 2), device=self.device)  # knee_pitch
            dof_pos[:, [4, 10]] = self.default_dof_pos[:, [
                4, 10]] + torch_rand_float(-0.1, 0.3, (len(env_ids), 2), device=self.device)  # ankle_pitch
            dof_pos[:, [5, 11]] = self.default_dof_pos[:, [
                5, 11]] + torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device)  # ankle_roll
            self.dof_pos[env_ids] = dof_pos[:]
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
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
            self.root_states[env_ids,
                             :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(
            env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        # base orientation
        base_orien_scale = self.cfg.init_state.init_base_angle_max
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), device=self.device).view(-1),
                                                             torch_rand_float(-base_orien_scale, base_orien_scale, (len(
                                                                 env_ids), 1), device=self.device).view(-1),
                                                             torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), device=self.device).view(-1))
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_dofs_gail(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: expert data frames to initialize motion with
        """
        if self.cfg.env.num_states == 31: # contains quaternion, base angular velocity
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 7:19], dtype=torch.float).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 19:31], dtype=torch.float).cuda()
        elif self.cfg.env.num_states == 27: # contains base angular velocity
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 3:15], dtype=torch.float).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 15:27], dtype=torch.float).cuda()
        elif self.cfg.env.num_states == 24: # only contains joint angles and velocities
            self.dof_pos[env_ids] = torch.tensor(
                frames[:, 0:12], dtype=torch.float).cuda()
            self.dof_vel[env_ids] = torch.tensor(
                frames[:, 12:24], dtype=torch.float).cuda()
        elif self.cfg.env.num_states == 20: # exclude hip roll
            dof_pos = torch.zeros((len(env_ids), self.num_dof),
                                  dtype=torch.float, device=self.device)
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [1, 7]]
            dof_pos[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                frames[:, 0:10], dtype=torch.float).cuda()
            self.dof_pos[env_ids, :] = dof_pos[:]
            dof_vel = torch.zeros((len(env_ids), self.num_dof),
                                  dtype=torch.float, device=self.device)
            dof_vel[:, [1, 7]] = 0.0
            dof_vel[:, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]] = torch.tensor(
                frames[:, 10:20], dtype=torch.float).cuda()
            self.dof_vel[env_ids, :] = dof_vel[:]
        elif self.cfg.env.num_states == 16: # exclude hip roll and yaw
            dof_pos = torch.zeros((len(env_ids), self.num_dof),
                                  dtype=torch.float, device=self.device)
            dof_pos[:, [1, 7]] = self.default_dof_pos[:, [1, 7]]
            dof_pos[:, [2, 8]] = self.default_dof_pos[:, [2, 8]] + torch_rand_float(
                -0.3, 0.3, (len(env_ids), 2), device=self.device)
            dof_pos[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                frames[:, 0:8], dtype=torch.float).cuda()
            self.dof_pos[env_ids, :] = dof_pos[:]
            dof_vel = torch.zeros((len(env_ids), self.num_dof),
                                    dtype=torch.float, device=self.device)
            dof_vel[:, [1, 7]] = 0.0
            dof_vel[:, [2, 8]] = 0.0
            dof_vel[:, [0, 3, 4, 5, 6, 9, 10, 11]] = torch.tensor(
                frames[:, 8:16], dtype=torch.float).cuda()
            self.dof_vel[env_ids, :] = dof_vel[:]
            
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_gail(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # ignore reference orientation
        self.root_states[env_ids, 3:7] = torch.zeros((len(env_ids), 4), 
                dtype=torch.float, device=self.device)  
        self.root_states[env_ids, 6] = 1.0  # set w component of quaternion to 1.0
        # base velocities
        self.root_states[env_ids, 7:10] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.env.num_states == 31: # contains quaternion, base angular velocity
            self.root_states[env_ids, 10:13] = torch.tensor(
                frames[:, 4:7], dtype=torch.float).cuda()
        elif self.cfg.env.num_states == 27: # contains base angular velocity
            self.root_states[env_ids, 10:13] = torch.tensor(
                frames[:, 0:3], dtype=torch.float).cuda()
        else: # only contains joint angles and velocities
            self.root_states[env_ids, 10:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3),
                                                               device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids])
        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        # randomize_ctrl_delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(
            self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        # set small commands to zero
        self.commands[:, 2] *= (torch.abs(self.commands[:, 2]) > 0.1).float()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.rand_push_force[:, :2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 7:9] = self.rand_push_force[:, :2] # set random base velocity in xy plane
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _refresh_rigid_body_states(self):
        # Periodic Reward Framework
        # refresh the states of the rigid bodies
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0.0  # commands
        noise_vec[3:15] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[15:27] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[27:39] = 0.  # previous actions
        noise_vec[39:42] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[42:45] = noise_scales.gravity * noise_level
        noise_vec[45:47] = 0.  # clock inputs
        noise_vec[47:49] = 0.  # phase ratio
        if self.cfg.terrain.measure_heights:
            noise_vec[49:239] = noise_scales.height_measurements * \
                noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(
            self.sim)  # Periodic Reward Framework
        self.gym.refresh_rigid_body_state_tensor(
            self.sim)  # Periodic Reward Framework

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13)  # Periodic Reward Framework
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
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

    def _create_envs(self):
        super()._create_envs()
        # knee_dof_indices: [3,9]
        # Periodic Reward Framework. distinguish between 4 feet
        for i in range(len(self.feet_names)):
            if "Lleg" in self.feet_names[i]:
                self.foot_index_left = self.feet_indices[i]
            elif "Rleg" in self.feet_names[i]:
                self.foot_index_right = self.feet_indices[i]
        
        self.knee_indices = torch.zeros(len(self.knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.knee_names[i])

    # ================================================ Private Functions ================================================== #

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_vel[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_pos[:, :, 2] < self.cfg.rewards.about_landing_threshold) & (
            ~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
    
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        # print(f"foot_dist: {foot_dist}")
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_body_states[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_clearance(self):
        '''reward for foot clearance'''
        foot_vel_xy_norm = torch.norm(self.foot_vel[:, :, [0, 1]], dim=-1)
        reward = torch.sum(foot_vel_xy_norm * torch.square(
            self.foot_pos[:, :, 2] - self.cfg.rewards.foot_clearance_target - self.cfg.rewards.foot_height_offset), dim=-1)
        # positive formulation can learn better
        return torch.exp(-reward / 0.01)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_dof_pos_limits(self):
        dof_indices_excluding_knee = [
            0, 1, 2, 4, 5, 6, 7, 8, 10, 11]  # exclude knee joints
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, dof_indices_excluding_knee] -
                          self.dof_pos_limits[dof_indices_excluding_knee, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos[:, dof_indices_excluding_knee] -
                          self.dof_pos_limits[dof_indices_excluding_knee, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
