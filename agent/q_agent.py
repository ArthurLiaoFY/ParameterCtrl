from collections import defaultdict

import numpy as np
from tensordict import TensorDict

from agent.agent import RLAgent


class QAgent(RLAgent):
    def __init__(self, state_dim, action_dim, **kwargs) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.__dict__.update(kwargs)
        self.start_explore

        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))

    def select_action(self, state: tuple) -> int:
        if self.explore and np.random.rand() <= self.explore_rate:
            action = np.random.choice(self.action_dim)

        else:
            action = np.argmax(self.q_table.get(state))
        return action

    def update_policy(self, sample_batch: TensorDict) -> None:
        for state, action, reward, next_state in zip(
            sample_batch["state"].tolist(),
            sample_batch["action"].tolist(),
            sample_batch["reward"].tolist(),
            sample_batch["next_state"].tolist(),
        ):
            td_target = (
                reward
                + self.discount_factor
                * self.q_table[tuple(next_state)][
                    np.argmax(self.q_table[tuple(next_state)])
                ]
            )

            td_error = td_target - self.q_table[tuple(state)][int(action[0])]
            self.q_table[tuple(state)][int(action[0])] += self.learning_rate * td_error

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min, self.learning_rate * self.learning_rate_decay_factor
        )

    def update_er(self, episode: int = 0) -> None:
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

    def save_table(
        self,
        model_file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        table_name: str = "q_table",
    ) -> None:
        np.save(
            f"{model_file_path}/{prefix}{table_name}{suffix}.npy",
            np.array(dict(self.q_table)),
        )

    def load_table(
        self,
        model_file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        table_name: str = "q_table",
    ) -> None:

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_dim),
            np.load(
                f"{model_file_path}/{prefix}{table_name}{suffix}.npy", allow_pickle=True
            ).item(),
        )
