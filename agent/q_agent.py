from collections import defaultdict

import numpy as np

from agent.agent import RLAgent


class Agent(RLAgent):
    def __init__(self, state_dim, action_dim, **kwargs) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.__dict__.update(kwargs)
        self.start_explore

        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))

    def select_action_idx(self, state: tuple) -> int:
        if self.explore and np.random.rand() <= self.explore_rate:
            action_idx = np.random.choice(self.action_dim)

        else:
            action_idx = np.argmax(self.q_table.get(state))
        return action_idx

    def action_idx_to_action(self, action_idx: int) -> tuple:
        return self.action_mapping_dict.get(action_idx)

    def select_action(self, state: tuple):
        action_idx = self.select_action_idx(state=state)
        return self.action_idx_to_action(action_idx=action_idx)

    def update_policy(
        self,
        state_tuple: tuple,
        action_idx: int,
        reward: float,
        next_state_tuple: tuple,
    ) -> None:
        td_target = (
            reward
            + self.discount_factor
            * self.q_table[next_state_tuple][np.argmax(self.q_table[next_state_tuple])]
        )

        td_error = td_target - self.q_table[state_tuple][action_idx]
        self.q_table[state_tuple][action_idx] += self.learning_rate * td_error

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min, self.learning_rate * self.learning_rate_decay
        )

    def update_er(self, episode: int = 0) -> None:
        if episode > self.fully_explore_step:
            self.explore_rate = max(
                self.explore_rate_min, self.explore_rate * self.explore_rate_decay
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
