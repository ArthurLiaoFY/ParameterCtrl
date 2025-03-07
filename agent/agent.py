from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state: torch.Tensor): ...

    @abstractmethod
    def update_policy(self, sample_batch: TensorDict): ...

    def update_lr(self) -> None: ...

    @property
    @abstractmethod
    def shutdown_explore(self) -> None: ...

    @property
    @abstractmethod
    def start_explore(self) -> None: ...

    @abstractmethod
    def save_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ): ...

    @abstractmethod
    def load_network(
        self,
        model_file_path: str,
        prefix: str,
        suffix: str,
    ): ...
