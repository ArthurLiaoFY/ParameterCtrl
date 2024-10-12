# %%
import torch
from torchrl.data import ReplayBuffer


class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
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


class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
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
            torch.cat(
                tensors=[state, action],
                dim=1,
            ),
        )


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.actor_head = torch.tanh(torch.nn.Linear(256, action_dim))
        self.critic_head = torch.nn.Linear(256, 1)

    def actor_forward(self, state, action):
        return self.actor_head(
            self.backbone(torch.cat([state, action], 1)),
        )

    def critic_forward(self, state, action):
        return self.critic_head(
            self.backbone(torch.cat([state, action], 1)),
        )
