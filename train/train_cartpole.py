# %%
import numpy as np
import torch

from agent.agent import RLAgent
from train.collect_buffer_data import CollectBufferData


class TrainAgent:
    def __init__(self, env, agent: RLAgent, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = env
        self.buffer_data = CollectBufferData(env=self.env, **kwargs)
        self.agent = agent

        self.max_train_reward = -np.inf

        self.episode_reward_traj = []

    def inference_once(self, episode: int):
        self.env.reset()
        inference_reward = 0
        cnt = 0

        self.agent.shutdown_explore

        for step in range(self.step_per_episode):
            action = (
                0
                if self.agent.select_action(
                    state=torch.Tensor(tuple(v for v in self.env.state.values()))
                )
                < 0.5
                else 1
            )
            step_loss = self.env.step(action=action)
            inference_reward += step_loss
            if step_loss <= self.step_loss_tolerance:
                cnt += 1
            else:
                cnt = 0

            if cnt == self.early_stop_patience:
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
            episode_loss = 0
            current_state_tensor = torch.Tensor(
                tuple(v for v in self.env.state.values())
            )
            cnt = 0
            for step in range(self.step_per_episode):
                # select action
                action = (
                    0
                    if self.agent.select_action(state=current_state_tensor) < 0.5
                    else 1
                )

                step_loss = self.env.step(action=action)
                if step_loss <= self.step_loss_tolerance:
                    cnt += 1
                else:
                    cnt = 0
                episode_loss += step_loss

                next_state_tensor = torch.Tensor(
                    tuple(v for v in self.env.state.values())
                )
                self.buffer_data.extend_buffer_data(
                    state=current_state_tensor[None, :],
                    action=torch.Tensor(action)[None, :],
                    reward=torch.Tensor([step_loss])[None, :],
                    next_state=next_state_tensor[None, :],
                )
                current_state_tensor = next_state_tensor

                # Sample a random mini-batch of N transitions (si, ai, ri, si+1) from R
                sample_batch = self.buffer_data.sample_buffer_data(
                    size=self.agent_kwargs.get("batch_size")
                )
                self.agent.update_policy(sample_batch)

                # self.actor_loss_history.append(actor_loss.detach().numpy().item())
                # self.critic_loss_history.append(critic_loss.detach().numpy().item())

                if cnt == self.early_stop_patience:
                    break

            print(f"episode [{episode}]-------------------------------------------")
            print(f"episode loss : {round(episode_loss, ndigits=4)}")
            print(f"jitter noise : {round(self.agent.jitter_noise, ndigits=4)}")
            print(f"learning rate : {round(self.agent.learning_rate, ndigits=4)}")
            self.episode_reward_traj.append(episode_loss)
            self.agent.update_lr()
            if episode % self.inference_each_k_episode == 0:
                self.inference_once(episode)
                if save_traj_to_buffer:
                    self.buffer_data.save_replay_buffer()

        if save_network:
            self.agent.save_network()
