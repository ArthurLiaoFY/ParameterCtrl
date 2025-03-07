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
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
        )
        # target net
        self.delay_actor = Actor(
            self.state_dim,
            self.action_dim,
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.learning_rate, amsgrad=True
        )
        self.delay_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, state: torch.Tensor):
        if np.random.rand() < self.explore_rate or self.explore:
            with torch.no_grad():
                action = np.argmax(self.actor(state).detach().numpy())
            self.explore_rate = max(
                self.explore_rate_min,
                self.explore_rate * self.explore_rate_decay_factor,
            )
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def update_policy(self, sample_batch: TensorDict) -> None:
        with torch.no_grad():
            td_target = sample_batch.get(
                "reward"
            ) + self.discount_factor * self.delay_actor(
                sample_batch.get("next_state"),
            )
        current_reward = self.actor(sample_batch.get("state"))
        critic_loss = torch.nn.functional.huber_loss(current_reward, td_target)

        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for actor, delay_actor in zip(
                self.actor.parameters(), self.delay_actor.parameters()
            ):
                delay_actor.data.copy_(
                    ((1 - self.tau) * delay_actor.data) + self.tau * actor.data
                )

        return None

    def update_lr(self) -> None:
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
