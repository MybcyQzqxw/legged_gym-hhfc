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
import csv
import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, GAIL
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, DiscriminatorNN
from rsl_rl.env import VecEnv
from rsl_rl.utils import Nomalizer

'''
name: class OnPolicyRunner
- 学习时调用这个函数，关键是 learn()
'''


class GAILRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.disc_cfg = train_cfg["discriminator"]
        self.device = device
        self.env = env
        self.num_step = 4
        self.num_state = self.cfg["num_state"]
        self.ref_traj = self.cfg["ref_traj"]
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)
        if env.cfg.env.is_amp:
            disc: DiscriminatorNN = DiscriminatorNN(self.num_step*self.num_state, 1,
                                                    **self.disc_cfg).to(self.device)
        else:
            disc: DiscriminatorNN = None

        # gail_nomalizer = Nomalizer(self.num_state*self.num_step)
        gail_nomalizer = Nomalizer(self.num_state)
        # gail_nomalizer = None

        self.alg: GAIL = GAIL(actor_critic, disc, gail_nomalizer, self.num_state, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions])
        # init sample storage
        self.alg.init_sample_storage(self.env.num_envs, self.num_step, self.num_step, 
                                     self.num_state, self.ref_traj)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # storage policy_state_buf; 30 is the number of state (quat, lin_vel, ang_vel, joint angle and velocities...)
        # self.step = 0
        self.policy_state_buf = torch.zeros(self.num_step, self.env.num_envs, self.num_state, device=self.device)
        _, _ = self.env.reset()


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
            name: learn()
            parameters：
                param1: num_learning_iterations, 学习的迭代次数
                param2: init_at_random_ep_len, bool, 是否随机初始化episode_length
            - 进行强化学习更新迭代，会调用算法 alg
            """
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        episode_sums = {'style': torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device, requires_grad=False)}
        d_x_buf = []

        # 每一次进行num_learning_iterations次迭代，总迭代次数如下
        # 每次与算法交互后，先迭代num_learning_iterations次，再更新discriminator和policy
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    # 与环境交互之后会得到obs, rewards, dones, infos
                    obs, privileged_obs, rewards, dones, infos, policy_state = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    # # 更新policy_state buffer，記錄策略的每一個時刻的狀態
                    self.alg.sample_state(policy_state)
                    self.add_policy_state(policy_state)
                    policy_state_buf = self.policy_state_buf.permute(1,0,2).reshape(-1,self.num_state*self.num_step)

                    if self.alg.gail_normalizer is not None:
                        with torch.no_grad():
                            end_step = 0
                            for i in range(self.num_step):
                                start_step = end_step
                                end_step += self.num_state
                                policy_state_buf[:, start_step:end_step] = self.alg.gail_normalizer.normalize_torch(
                                    policy_state_buf[:, start_step:end_step], self.device)

                    if self.env.cfg.env.is_amp:
                        # policy_state_buf = self.alg.gail_normalizer.normalize_torch(policy_state_buf, self.device)
                        self.alg.disc.eval()
                        d_x = self.alg.dis(policy_state_buf)
                        # dt * weight * equation
                        r_s = 0.02 * 5.0 * (torch.max(torch.zeros_like(d_x), 1-0.25*torch.square(d_x-1)).flatten())
                    else:
                        r_s = 0

                    rewards += r_s

                    # rew_style buf
                    episode_sums["style"] += r_s
                    env_ids = torch.nonzero(dones, as_tuple=True)
                    infos["episode"]["rew_style"] = torch.mean(episode_sums["style"][env_ids])/self.env.max_episode_length_s
                    episode_sums["style"][env_ids] = 0.
                    if torch.sum(torch.isnan(infos["episode"]["rew_style"])) > 0:
                        infos["episode"]["rew_style"] = torch.zeros(1).to(self.device)

                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            if self.env.cfg.env.is_amp:
                # update得到平均的disc_loss(discriminator)
                self.alg.disc.train()
                mean_disc_loss = self.alg.disc_update()
            else:
                mean_disc_loss = 0

            # update可以得到平均的value_loss & surrogate_loss(policy)
            mean_value_loss, mean_surrogate_loss = self.alg.gen_update()
            stop = time.time()
            learn_time = stop - start
            ''
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def add_policy_state(self, policy_state):
        temp = self.policy_state_buf.clone()
        self.policy_state_buf[:-1,:,:] = temp[1:,:,:]
        self.policy_state_buf[-1,:,:] = policy_state

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/dics_loss',locs['mean_disc_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if self.env.cfg.env.is_amp:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'discriminator_state_dict': self.alg.disc.state_dict(),
                'gail_normalizer': self.alg.gail_normalizer,
                'iter': self.current_learning_iteration,
                'infos': infos,
            }, path)
        else:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'gail_normalizer': self.alg.gail_normalizer,
                'iter': self.current_learning_iteration,
                'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.disc.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.gail_normalizer = loaded_dict['gail_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def disc_reward_origin(self, D_x):
        """Converts the discriminator logits raw value to a reward signal.

            The GAIL paper defines the cost function of the generator as:

            .. math::

                \log{D}

            as shown on line 5 of Algorithm 1. In the paper, :math:`D` is the probability
            distribution learned by the discriminator, where :math:`D(X)=1` if the trajectory
            comes from the generator, and :math:`D(X)=0` if it comes from the expert.
            In this implementation, we have decided to use the opposite convention that
            :math:`D(X)=0` if the trajectory comes from the generator,
            and :math:`D(X)=1` if it comes from the expert. Therefore, the resulting cost
            function is:

            .. math::

                \log{(1-D)}

            Since our algorithm trains using a reward function instead of a loss function, we
            need to invert the sign to get:

            .. math::

                R=-\log{(1-D)}=\log{\frac{1}{1-D}}

                R=-\log{(1-D)}=\log{\frac{D}{1-D}}
            Now, let :math:`L` be the output of our reward net, which gives us the logits of D
            (:math:`L=\operatorname{logit}{D}`). We can write:

            """
        return torch.log(D_x/(1-D_x))
