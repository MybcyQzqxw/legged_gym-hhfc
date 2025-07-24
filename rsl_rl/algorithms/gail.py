import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd

import copy

from rsl_rl.modules import ActorCritic
from rsl_rl.modules import DiscriminatorNN
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage import SampleStorage


BATCH_SIZE = 64

''' GAIL: Generative Adversarial Imitation Learning'''
class GAIL:
    """
    Generative Adversarial Imitation Learning

    Main idea of the algorithm:

    """
    actor_critic: ActorCritic
    disc: DiscriminatorNN
    def __init__(self,
                 actor_critic,
                 disc,
                 gail_normalizer,
                 num_state,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.epoch = 4

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.sample_transition = SampleStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Create graph for optimising recording states on discriminator
        self.disc = disc
        if self.disc is not None:
            self.disc.to(self.device)
            disc_params = [
                {'params': self.disc.trunk.parameters(),
                 'weight_decay': 10e-4, 'name': 'disc_trunk'},
                {'params': self.disc.disc_linear.parameters(),
                 'weight_decay': 10e-2, 'name': 'disc_head'}]
            self.disc_optimizer = optim.Adam(disc_params, lr=1e-3)

        self.gail_normalizer = gail_normalizer
        self.num_state = num_state

        self.sample_storage = None

        self.loss_fn = get_adversarial_losses_fn('lsgan')


    def disc_update(self):
        loss = 0.0
        num_step = self.sample_storage.num_step
        expert_rollout = self.sample_storage.expert_observation
        policy_rollout = self.sample_storage.policy_state[:num_step,:,:]

        for _ in range(self.epoch):
            # generate n=BATCH_SIZE random indices for policy data
            policy_indices = torch.tensor(random.sample(range(policy_rollout.shape[1]), BATCH_SIZE), device=self.device)

            # generate n=BATCH_SIZE random indices for expert data
            expert_indices = torch.randperm(expert_rollout.size(0)-num_step)[:BATCH_SIZE]

            # initialize empty tensor to storage the selected rows (batch)
            # expert_batch = torch.empty(0, expert_rollout.size(1), dtype=expert_rollout.dtype)
            expert_batch_list = []

            # select num_step of data starting from each random index
            for index in expert_indices:
                expert_selected = expert_rollout[index:index+num_step, :].reshape(1,-1)
                expert_batch_list.append(expert_selected)

            expert_batch = torch.cat((expert_batch_list)).to(torch.float32)

            # select from each random index and flatten dimension
            policy_batch = policy_rollout[:,policy_indices,:].to(torch.float32)
            policy_batch = policy_batch.permute(1,0,2).reshape(BATCH_SIZE,-1)

            if self.gail_normalizer is not None:
                with torch.no_grad():
                    end_step = 0
                    for i in range(num_step):
                        start_step = end_step
                        end_step += self.num_state
                        expert_batch[:,start_step:end_step] = self.gail_normalizer.normalize_torch(expert_batch[:, start_step:end_step], self.device)
                        policy_batch[:,start_step:end_step] = self.gail_normalizer.normalize_torch(policy_batch[:, start_step:end_step], self.device)

            # disc_input = torch.cat([expert_batch, policy_batch]).to(torch.float)
            expert_d = self.disc(expert_batch).flatten()
            policy_d = self.disc(policy_batch).flatten()

            # 加入梯度惩罚
            grad_pen = self.compute_grad_pen(expert_batch, policy_batch)

            expert_loss, policy_loss = self.loss_fn(expert_d, policy_d)
            disc_loss = expert_loss + policy_loss + grad_pen

            loss += disc_loss.item()

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            if self.gail_normalizer is not None:
                # a = policy_batch.cpu().numpy()
                self.gail_normalizer.update(policy_batch[:,:self.num_state].cpu().numpy())
                self.gail_normalizer.update(expert_batch[:,:self.num_state].cpu().numpy())

        # clear the sample storage: raise AssertionError("Samples Rollout buffer overflow")
        self.sample_storage.clear()

        return loss/self.epoch

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    def init_sample_storage(self, num_envs, num_transitions_per_env, num_step, 
                            num_state, ref_traj):
        self.sample_storage = SampleStorage(num_envs, num_transitions_per_env, num_step, 
                                            num_state, ref_traj, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def dis(self, joint_state):
        self.sample_transition.dis = self.disc(joint_state).detach()
        return self.sample_transition.dis


    def sample_state(self, state):
        # need: the orientation of base (euler angle), the linear and angular velocities of base
        # and the position and velocities of joint
        self.sample_transition.policy_state = state

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        # Record the sample transition
        self.sample_storage.add_transitions(self.sample_transition)
        self.sample_transition.clear()

    # 计算return, 在rollout_storage.py, 计算了折扣回报和advantages
    # rollout_storage.py
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # 更新policy network
    def gen_update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            # 动作概率分布的log，value的值，动作的均质和方差
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL 散度的计算，改变学习率，防止importance sample的两个概率相差太大
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss 计算actor的loss
            # PPO算法的核心部分 ratio就是算两个action_prob之间的比率（不能差太大）
            # ratio*advantage的期望就是用于梯度的loss_function，但是需要clip限幅
            # actor网络需要输出的动作优势尽可能的大
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            # value的估计值和回报之间作差 MSE 均方差
            # target_value = returns = advantages + values (参考CSDN, TD)
            # TD_error = value - returns, loss = TD_error^2
            if self.use_clipped_value_loss:
                # 如果是value function loss clipping (效果一般) 则有
                # L_v = max[(value(t)-return)^2, ((value(t-1) + clip(value(t)-value(t-1)，-epsilon,epsilon))-return)^2]
                # clipped_TD_error = clipped_value - returns
                # loss = max[TD_error^2, clipped_TD_error^2]
                # 在这里面 target_values_batch 应该表示的是前一个时刻的预测值？ value(t-1)
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # actor-critic共享一个网络，计算最终的损失函数，熵损失entropy鼓励actor探索更多的行为（根据概率分布计算策略的熵)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step 梯度上升
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

    def logits_expert_is_high(self,
                              state: torch.Tensor,
                              action: torch.Tensor,
                              next_state: torch.Tensor,
                              done: torch.Tensor):
        """Compute the discriminator's logits for each state-action sample ?

        A high value corresponds to predicting expert, and a low value corresponds yo predicting generator.

        Args:
             state:
             action:
             next_state:
             done:

        Returns:
            Discriminator logits of shape 'batch_size'. A high output indicates an expert-like transition.

        todo:human demonstration only includes state
        """
        logits = state

    def compute_grad_pen(self, expert_data, policy_data, lambda_=5):
        alpha = torch.rand(expert_data.size(0), 1).to(expert_data.device)
        # alpha = alpha.expend_as(expert_data).to(expert_data.device)
        alpha = torch.ones_like(alpha)

        mixedup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixedup_data.requires_grad = True

        disc = self.disc(mixedup_data).flatten()
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixedup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


def get_gan_losses_fun():
    bce = torch.nn.BCEWithLogitsLoss()

    def dis_loss_fn(expert_logit, policy_logit):
        expert_loss = bce(expert_logit, torch.ones_like(expert_logit))
        policy_loss = bce(policy_logit, torch.zeros_like(policy_logit))
        return expert_loss, policy_loss

    return dis_loss_fn

def get_lsgan_losses_fun():
    mse = torch.nn.MSELoss()

    def dis_loss_fn(expert_logit, policy_logit):
        """
        the loss function is similar to AMP(Peng)
        E_dM((D(s,s')-1)^2) + E_dpi((D(s,s')+1)^2)
        """
        expert_loss = mse(expert_logit, torch.ones_like(expert_logit))
        policy_loss = mse(policy_logit, -torch.ones_like(policy_logit))
        return expert_loss, policy_loss

    return dis_loss_fn

def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fun()
    elif mode =='lsgan':
        return get_lsgan_losses_fun()


