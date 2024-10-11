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


# %%
class DDPG:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discount: float = 0.99,
        tau: float = 0.001,
    ) -> None:
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.actor_prime = Actor(state_dim, action_dim)
        self.actor_prime.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_prime = Critic(state_dim, action_dim)
        self.critic_prime.load_state_dict(self.critic.state_dict())

        self.discount = discount
        self.tau = tau

        self.actor_loss = []
        self.critic_loss = []

    def train(self, replay_buffer: ReplayBuffer, batch_size=256):
        # Sample a random mini-batch of batch size from ReplayBuffer
        state, action, next_state, reward = replay_buffer.sample(batch_size)

        # Set yi
        y = reward + self.discount * self.critic_prime(
            next_state,
            self.actor_prime(next_state),
        )
        print(y.shape)

        # update critic by mse loss

        # update actor by

        # update the target networks
