import torch


class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Actor, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_dim),
        )

    def forward(self, state):
        return torch.tanh(self.actor(state))
