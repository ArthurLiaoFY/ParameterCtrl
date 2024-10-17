import numpy as np
import torch
from tensordict import TensorDict


class DeepQNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(DeepQNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nn.Tanh()(self.model(state))


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

    def select_action_idx(self, state_tuple: tuple) -> int:
        if self.explore and np.random.uniform(low=0, high=1) <= self.explore_rate:
            action_idx = np.random.choice(self.action_dim)

        else:
            action_idx = np.argmax(
                self.dqn(torch.Tensor(state_tuple)[None, :]).detach().numpy(),
                axis=1,
            ).squeeze()
        return action_idx

    def update_policy(
        self,
        episode: int,
        sample_batch: TensorDict,
    ) -> None:
        action_logit = self.dqn(sample_batch.get("state"))
        next_action_logit = self.dqn_prime(sample_batch.get("next_state"))

        td_target = (
            sample_batch.get("reward")
            + self.discount_factor
            * torch.max(next_action_logit, dim=1, keepdim=True).values
        )

        dqn_loss = torch.nn.functional.mse_loss(
            td_target,
            torch.gather(
                input=action_logit,
                index=sample_batch.get("action_idx").type(torch.int64),
                dim=1,
            ),
        )
        self.dqn.zero_grad()
        dqn_loss.backward()
        self.dqn_optimizer.step()

    def update_lr_er(self, episode: int = 0) -> None:
        if episode > self.fully_explore_step:
            self.explore_rate = max(
                self.explore_rate_min, self.explore_rate * self.explore_rate_decay
            )
        self.learning_rate = max(
            self.learning_rate_min, self.learning_rate * self.learning_rate_decay
        )

    @property
    def shutdown_explore(self) -> None:
        self.explore = False

    @property
    def start_explore(self) -> None:
        self.explore = True

    def save_network(
        self,
        model_file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        dqn_name: str = "dqn",
    ) -> None:
        torch.save(
            self.dqn.state_dict(),
            f"{model_file_path}/{prefix}{dqn_name}{suffix}.pt",
        )

    def load_network(
        self,
        model_file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        dqn_name: str = "dqn",
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
