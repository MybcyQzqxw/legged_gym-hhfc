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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

'''
name: class DiscriminatorNN
- create Disc NN network
'''


class DiscriminatorNN(nn.Module):

    def __init__(self, num_state,
                 num_output=1,
                 hidden_dims=[512, 256],
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(DiscriminatorNN, self).__init__()

        mlp_input_dim_a = num_state
        activation = nn.ReLU()
        # activation_end = nn.Tanh()

        # Discriminator 策略网络的搭建
        disc_layers = []
        curr_in_dim = mlp_input_dim_a
        for l in range(len(hidden_dims)):
            disc_layers.append(nn.Linear(curr_in_dim, hidden_dims[l]))
            disc_layers.append(activation)
            curr_in_dim = hidden_dims[l]
        self.trunk = nn.Sequential(*disc_layers)
        self.disc_linear = nn.Linear(hidden_dims[-1], 1)

        # disc_layers.append(nn.Linear(mlp_input_dim_a, hidden_dims[0]))
        # disc_layers.append(activation)
        # for l in range(len(hidden_dims)):
        #     if l == len(hidden_dims) - 1:
        #         disc_layers.append(nn.Linear(hidden_dims[l], num_output))
        #         # disc_layers.append(activation_end)
        #     else:
        #         disc_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        #         disc_layers.append(activation)
        # self.disc = nn.Sequential(*disc_layers)

        # print(f"Discriminator MLP: {self.disc}")


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self, x):
        h = self.trunk(x)
        d = self.disc_linear(h)
        return d

# 激活函数的设置
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
