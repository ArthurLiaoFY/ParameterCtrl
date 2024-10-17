import numpy as np
import torch
from tensordict import TensorDict


class DeepQNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(DeepQNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.model(state))


class DoubleDeepQNetwork:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.start_explore

        # actor network
        self.dqn = DeepQNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        # target network
        self.dqn_prime = DeepQNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        )
        self.load_network()

        self.dqn_optimizer = torch.optim.Adam(
            self.dqn.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, normed_state: torch.Tensor):
        if not self.explore:
            additional_noise = np.array([0.0 for _ in range(self.action_dim)])
        else:
            self.jitter_noise = max(
                self.jitter_noise_min,
                self.jitter_noise * self.jitter_noise_decay_factor,
            )
            additional_noise = np.random.randn() * self.jitter_noise

        return self.dqn(normed_state).detach().numpy() + additional_noise

    def update_policy(
        self,
        sample_batch: TensorDict,
    ) -> None:

        td_target = sample_batch.get("reward")[
            :, None
        ] + self.discount_factor * self.dqn_prime(sample_batch.get("next_normed_state"))

        dqn_loss = torch.nn.functional.mse_loss(
            td_target,
            self.dqn(sample_batch.get("normed_state")),
        )

        self.dqn.zero_grad()
        dqn_loss.backward()
        self.dqn_optimizer.step()

        with torch.no_grad():
            for dqn, dqn_prime in zip(
                self.dqn.parameters(), self.dqn_prime.parameters()
            ):
                dqn_prime.data.copy_(
                    ((1 - self.tau) * dqn_prime.data) + self.tau * dqn.data
                )

        return dqn_loss

    def update_lr(self, episode: int = 0) -> None:
        self.learning_rate = max(
            self.learning_rate_min,
            self.learning_rate * self.learning_rate_decay_factor,
        )

    @property
    def shutdown_explore(self) -> None:
        self.explore = False

    @property
    def start_explore(self) -> None:
        self.explore = True

    def save_network(
        self,
        model_file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        dqn_name: str = "dqn_network",
    ) -> None:
        torch.save(
            self.dqn.state_dict(),
            f"{model_file_path}/{prefix}{dqn_name}{suffix}.pt",
        )

    def load_network(
        self,
        model_file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        dqn_name: str = "dqn_network",
    ) -> None:
        try:
            self.dqn.load_state_dict(
                torch.load(
                    f"{model_file_path}/{prefix}{dqn_name}{suffix}.pt",
                    weights_only=True,
                )
            )
            self.dqn_prime.load_state_dict(self.dqn.state_dict())
            print(
                f"Found trained model under {model_file_path}, weights have been loaded."
            )
        except FileNotFoundError:
            pass
