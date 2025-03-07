import gymnasium as gym
import numpy as np


class CartPole:
    def __init__(self) -> None:
        self.env = gym.make("CartPole-v1")
        self.reset()

    def reset(self) -> None:
        state, _ = self.env.reset()
        self.state = dict(
            zip(
                ["CartPosition", "CartVelocity", "PoleAngle", "PoleAngularVelocity"],
                state,
            )
        )

    def step(self, action: tuple[float, float], return_xy: bool = False):
        state, reward, terminated, truncated, _ = self.env.step(action=action)
        dict(
            zip(
                ["CartPosition", "CartVelocity", "PoleAngle", "PoleAngularVelocity"],
                state,
            )
        )

        return reward
