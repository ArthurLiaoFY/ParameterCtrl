import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

# selection on which GPU code should be run..
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# if GPU available then use the GPU otherwise use the CPU.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# if torch.cuda.is_available():
# 	print("training on the nvidia GPU........")

# torch.cuda.empty_cache()

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):  # 給 distribution
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return torch.tanh(self.actor(state))


class Critic(nn.Module):  # 給評價值

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.critic(
            torch.cat([state, action], 1),
        )
