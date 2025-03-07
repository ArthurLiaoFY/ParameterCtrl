import numpy as np
import torch
from tensordict import TensorDict

from agent.agent import RLAgent
from agent.model.actor import Actor


class DeepQNetwork(RLAgent):
    def __init__(self, state_dim, action_dim, **kwargs) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.__dict__.update(**kwargs)
        self.start_explore
        # policy net
        self.q_network = Actor(
            self.state_dim,
            self.action_dim,
        )
        # target net
        self.delay_q_network = Actor(
            self.state_dim,
            self.action_dim,
        )
        self.q_network_optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=self.learning_rate, amsgrad=True
        )
        self.delay_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: tuple):
        if np.random.rand() < self.explore_rate or self.explore:
            with torch.no_grad():
                action = np.argmax(self.q_network(torch.Tensor(state)).detach().numpy())
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def update_policy(self, sample_batch: TensorDict) -> None:
        with torch.no_grad():
            td_target = sample_batch.get(
                "reward"
            ) + self.discount_factor * self.delay_q_network(
                sample_batch.get("next_state")
            )
        current_reward = self.q_network(sample_batch.get("state"))
        q_loss = torch.nn.functional.huber_loss(current_reward, td_target)

        self.q_network_optimizer.zero_grad()
        q_loss.backward()
        self.q_network_optimizer.step()

        with torch.no_grad():
            for q_network, delay_q_network in zip(
                self.q_network.parameters(), self.delay_q_network.parameters()
            ):
                delay_q_network.data.copy_(
                    ((1 - self.tau) * delay_q_network.data) + self.tau * q_network.data
                )

        return q_loss

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min,
            self.learning_rate * self.learning_rate_decay_factor,
        )

    def update_er(self, episode: int) -> None:
        if episode > self.fully_explore_step:
            self.explore_rate = max(
                self.explore_rate_min,
                self.explore_rate * self.explore_rate_decay_factor,
            )

    @property
    def shutdown_explore(self) -> None:
        self.explore = False

    @property
    def start_explore(self) -> None:
        self.explore = True

    def save_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ):
        pass

    def load_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ):
        pass
