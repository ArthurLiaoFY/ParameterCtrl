# %%
import numpy as np
import torch

from agent.agent import RLAgent
from train.collect_buffer_data import CollectBufferData
from utils.plot_f import plot_inference_result, plot_reward_trend


class TrainCSTR:
    def __init__(self, env, agent: RLAgent, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = env
        self.buffer_data = CollectBufferData(env=self.env, **kwargs)
        self.agent = agent

        self.max_train_reward = -np.inf

        self.episode_reward_traj = []
        self.actor_loss_history = []
        self.critic_loss_history = []

        self.inference_traj = {
            "ideal_Ca": self.cstr_env_kwargs.get("ideal_Ca"),
            "ideal_Cb": self.cstr_env_kwargs.get("ideal_Cb"),
            "ideal_Tr": self.cstr_env_kwargs.get("ideal_Tr"),
            "ideal_Tk": self.cstr_env_kwargs.get("ideal_Tk"),
            "Ca": {},
            "Cb": {},
            "Tr": {},
            "Tk": {},
            "F": {},
            "Q": {},
        }

    def inference_once(self, episode: int):
        self.env.reset()
        inference_reward = 0
        cnt = 0

        self.agent.shutdown_explore

        for step in range(self.step_per_episode):
            normed_action = self.agent.select_action(
                state=torch.Tensor(tuple(v for v in self.env.normed_state.values()))
            )
            step_loss = self.env.step(
                action=self.env.revert_normed_action(normed_action=normed_action)
            )
            inference_reward += step_loss
            if step_loss <= self.step_loss_tolerance:
                cnt += 1
            else:
                cnt = 0

            if cnt == self.early_stop_patience:
                break

        self.inference_traj["Ca"][episode] = self.env.Ca_traj
        self.inference_traj["Cb"][episode] = self.env.Cb_traj
        self.inference_traj["Tr"][episode] = self.env.Tr_traj
        self.inference_traj["Tk"][episode] = self.env.Tk_traj
        self.inference_traj["F"][episode] = self.env.F_traj
        self.inference_traj["Q"][episode] = self.env.Q_traj

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
            current_normed_state_tensor = torch.Tensor(
                tuple(v for v in self.env.normed_state.values())
            )
            cnt = 0
            for step in range(self.step_per_episode):
                # select action
                normed_action = self.agent.select_action(
                    state=current_normed_state_tensor
                )

                step_loss = self.env.step(
                    action=self.env.revert_normed_action(normed_action=normed_action)
                )
                if step_loss <= self.step_loss_tolerance:
                    cnt += 1
                else:
                    cnt = 0
                episode_loss += step_loss

                next_normed_state_tensor = torch.Tensor(
                    tuple(v for v in self.env.normed_state.values())
                )
                self.buffer_data.extend_buffer_data(
                    state=current_normed_state_tensor[None, :],
                    action=torch.Tensor(normed_action)[None, :],
                    reward=torch.Tensor([step_loss])[None, :],
                    next_state=next_normed_state_tensor[None, :],
                )
                current_normed_state_tensor = next_normed_state_tensor

                # Sample a random mini-batch of N transitions (si, ai, ri, si+1) from R
                sample_batch = self.buffer_data.sample_buffer_data(
                    size=self.agent_kwargs.get("batch_size")
                )
                actor_loss, critic_loss = self.agent.update_policy(sample_batch)

                self.actor_loss_history.append(actor_loss.detach().numpy().item())
                self.critic_loss_history.append(critic_loss.detach().numpy().item())

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

        plot_inference_result(inference_traj=self.inference_traj)
        plot_reward_trend(rewards=self.episode_reward_traj)

        if save_network:
            self.agent.save_network()
