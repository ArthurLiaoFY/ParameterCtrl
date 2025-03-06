import torch


class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, state, action):
        return self.critic(
            torch.cat(
                tensors=[state, action],
                dim=1,
            ),
        )
