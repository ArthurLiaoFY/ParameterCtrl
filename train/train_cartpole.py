# %%
import numpy as np
import torch

from agent.agent import RLAgent
from train.collect_buffer_data import CollectBufferData


class TrainCartPole:
    def __init__(self, env, agent: RLAgent, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = env
        self.buffer_data = CollectBufferData(env=self.env, **kwargs)
        self.agent = agent

        self.max_train_reward = -np.inf

        self.episode_reward_traj = []
        self.q_loss_history = []

    def inference_once(self, episode: int):
        self.env.reset()
        inference_reward = 0

        self.agent.shutdown_explore

        for step in range(self.step_per_episode):
            action = self.agent.select_action(
                state=torch.Tensor(tuple(v for v in self.env.state.values()))
            )
            terminated, truncated = self.env.step(action=action)
            reward = 0 if terminated else 1
            inference_reward += reward
            if terminated or truncated:
                break

        # restart explore
        self.agent.start_explore

    def train_online_agent(
        self,
        save_traj_to_buffer: bool = True,
        save_network: bool = True,
    ):
        self.agent.start_explore

        for episode in range(1, self.n_episodes + 1):
            self.env.reset()
            episode_reward = 0
            current_state_tensor = torch.Tensor(
                tuple(v for v in self.env.state.values())
            )
            for step in range(self.step_per_episode):
                # select action
                action = self.agent.select_action(
                    state=torch.Tensor(tuple(v for v in self.env.state.values()))
                )

                terminated, truncated = self.env.step(action=action)
                reward = 0 if terminated else 1
                episode_reward += reward

                # send to buffer
                next_state_tensor = torch.Tensor(
                    tuple(v for v in self.env.state.values())
                )

                self.buffer_data.extend_buffer_data(
                    state=current_state_tensor[None, :],
                    action=torch.Tensor([action])[None, :],
                    reward=torch.Tensor([reward])[None, :],
                    next_state=next_state_tensor[None, :],
                )
                current_state_tensor = next_state_tensor

                # Sample a random mini-batch of N transitions (si, ai, ri, si+1) from R
                sample_batch = self.buffer_data.sample_buffer_data(
                    size=self.agent_kwargs.get("batch_size")
                )
                q_loss = self.agent.update_policy(sample_batch)

                self.q_loss_history.append(q_loss)

                if terminated or truncated:
                    break

            print(f"episode [{episode}]-------------------------------------------")
            print(f"episode loss : {round(episode_reward, ndigits=4)}")
            print(f"jitter noise : {round(self.agent.jitter_noise, ndigits=4)}")
            print(f"explore rate : {round(self.agent.explore_rate, ndigits=4)}")
            print(f"learning rate : {round(self.agent.learning_rate, ndigits=4)}")
            self.episode_reward_traj.append(episode_reward)
            self.agent.update_lr()
            self.agent.update_er(episode=episode)
            if episode % self.inference_each_k_episode == 0:
                self.inference_once(episode)
                if save_traj_to_buffer:
                    self.buffer_data.save_replay_buffer()

        if save_network:
            self.agent.save_network()
