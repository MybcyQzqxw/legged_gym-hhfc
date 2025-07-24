import torch
import numpy as np
import pandas as pd
import os
import copy

from rsl_rl.utils import split_and_pad_trajectories

class SampleStorage:
    """
    rollout: the buffer of expert and policy

    form1: the buffer of the current kinematic state (including joint position and velocities)
    form2: the buffer of the state transition
    """
    class Transition:
        def __init__(self):
            self.policy_state = None
            self.dis = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, num_step, num_state, 
                 ref_traj, device='cpu'):

        self.device = device
        self.num_step = num_step
        self.num_state = num_state
        
        ref_df_data = np.array(pd.read_csv(ref_traj, sep='\t'))

        self.expert_observation = torch.tensor(ref_df_data, device=self.device)

        self.policy_state = torch.zeros(num_transitions_per_env, num_envs, self.num_state, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        # if self.step >= self.num_transitions_per_env:
        #     raise AssertionError("Samples Rollout buffer overflow")
        # self.policy_joint_state[self.step].copy_(transition.policy_joint_state)
        temp = self.policy_state.clone()
        self.policy_state[:-1,:,:] = temp[1:,:,:]
        self.policy_state[-1,:,:] = copy.copy(transition.policy_state)
        # self.policy_state[self.step].copy_(transition.policy_state)
        self.step += 1


    def clear(self):
        self.step = 0

