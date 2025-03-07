from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state: torch.Tensor): ...

    @abstractmethod
    def update_policy(self, sample_batch: TensorDict): ...

    @abstractmethod
    def update_lr(self) -> None: ...

    def update_er(self) -> None: ...

    @property
    @abstractmethod
    def shutdown_explore(self) -> None: ...

    @property
    @abstractmethod
    def start_explore(self) -> None: ...
