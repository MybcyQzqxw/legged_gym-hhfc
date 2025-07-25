# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, \
    export_policy_as_onnx, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1  # 5
    env_cfg.terrain.num_cols = 1  # 5
    env_cfg.terrain.terrain_length = 25.
    env_cfg.terrain.terrain_width = 25.
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]  # min max [m/s]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # exo_policy = ppo_runner.get_exo_inference_policy(device=env.device)
    # slnet = ppo_runner.slmodel

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    if EXPORT_AS_ONNX:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies', 'humanoid')
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, env_cfg, train_cfg)
        print('Exported policy as ONNX to: ', path)
    
    #---------- FOLLOW_ROBOT Camera Config ----------#
    camera_lookat_follow = np.array(env_cfg.viewer.lookat)
    camera_deviation_follow = np.array([-0.5, 2.0, 0.])
    camera_position_follow = camera_lookat_follow - camera_deviation_follow

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 3  # which joint is used for logging
    stop_state_log = 800  # number of steps before plotting states
    # number of steps before print average episode rewards
    stop_rew_log = env.max_episode_length + 1
    
    env._resample_commands([0])
    spd_change_interval = 300

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _ = env.step(actions.detach())
        
        #----- Speed Change -----#
        if i % spd_change_interval == 0 and i != 0:
            if env.command_ranges["lin_vel_x"][0] == 0.0:
                env.command_ranges["lin_vel_x"][0] = env.command_ranges["lin_vel_x"][1] = 1.2
                env._resample_commands(torch.arange(env.num_envs, device=env.device))
                print("Speed Change: 0.0 -> 1.2")
            elif env.command_ranges["lin_vel_x"][0] == 1.2:
                env.command_ranges["lin_vel_x"][0] = env.command_ranges["lin_vel_x"][1] = 2.0
                env._resample_commands(torch.arange(env.num_envs, device=env.device))
                print("Speed Change: 1.2 -> 2.0")
            elif env.command_ranges["lin_vel_x"][0] == 2.0:
                env.command_ranges["lin_vel_x"][0] = env.command_ranges["lin_vel_x"][1] = 0.0
                env._resample_commands(torch.arange(env.num_envs, device=env.device))
                print("Speed Change: 2.0 -> 0.0")

        if FOLLOW_ROBOT:
            # refresh where camera looks at(robot 0 base)
            camera_lookat_follow = env.root_states[robot_index, 0:3].cpu().numpy()
            # refresh camera's position
            camera_position_follow = camera_lookat_follow - camera_deviation_follow
            env.set_camera(camera_position_follow, camera_lookat_follow)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    # 'exo_tor_r': action_exo[0, 0].clone().detach().cpu().numpy(),
                    # 'exo_tor_l': action_exo[0, 1].clone().detach().cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_AS_ONNX = True
    FOLLOW_ROBOT = True
    # header_of_file()
    args = get_args()
    play(args)
