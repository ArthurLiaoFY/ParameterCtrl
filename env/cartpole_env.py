import gymnasium as gym
import numpy as np


class CartPole:
    def __init__(self) -> None:
        self.env = gym.make(id="CartPole-v1", max_episode_steps=200)
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
        state, reward, terminated, truncated, infos = self.env.step(action=action)
        self.state = dict(
            zip(
                ["CartPosition", "CartVelocity", "PoleAngle", "PoleAngularVelocity"],
                state,
            )
        )

        return terminated, truncated
