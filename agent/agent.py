from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class RLAgent(ABC):
    def __init__(self, **kwargs) -> None:
        self.start_explore
        self.__dict__.update(**kwargs)

    @abstractmethod
    def select_action(self, state: torch.Tensor):
        pass

    @abstractmethod
    def update_policy(self, sample_batch: TensorDict):
        pass

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min,
            self.learning_rate * self.learning_rate_decay_factor,
        )

    @property
    @abstractmethod
    def shutdown_explore(self) -> None:
        self.explore = False

    @property
    @abstractmethod
    def start_explore(self) -> None:
        self.explore = True

    @abstractmethod
    def save_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ):
        pass

    @abstractmethod
    def load_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ):
        pass
