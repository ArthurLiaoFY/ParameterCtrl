import gymnasium as gym
import numpy as np

from env.env import RLEnv


class CartPole(RLEnv):
    def __init__(self, seed: int | None = None, **kwargs) -> None:
        if seed is None:
            self.seed = np.random
        else:
            self.seed = np.random.RandomState(seed)
        self.__dict__.update(**kwargs)
        self.reset()
