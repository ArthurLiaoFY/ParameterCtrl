# %%
import numpy as np
import torch
from tensordict import TensorDict

from agent.ddpg import DDPG
from cstr_env import CSTREnv
from train.collect_buffer_data import CollectBufferData
from utils.plot_f import plot_inference_result, plot_reward_trend


class TrainDDPG:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)
        self.ddpg = DDPG(**self.ddpg_kwargs)

        self.max_total_reward = -np.inf

        self.episode_loss_traj = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def inference_once(self):
        self.env.reset()
        inference_reward = 0
        cnt = 0
        for step in range(self.step_per_episode):
            normed_action = self.ddpg.select_action(
                normed_state=torch.Tensor(
                    tuple(v for v in self.env.normed_state.values())
                )
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
        return (
            self.env.Ca_traj,
            self.env.Cb_traj,
            self.env.Tr_traj,
            self.env.Tk_traj,
            self.env.F_traj,
            self.env.Q_traj,
        )

    def train_agent(
        self,
        buffer_data: CollectBufferData,
        save_network: bool = True,
    ):
        inference_traj = {
            "ideal_Ca": self.env.ideal_Ca,
            "ideal_Cb": self.env.ideal_Cb,
            "ideal_Tr": self.env.ideal_Tr,
            "ideal_Tk": self.env.ideal_Tk,
            "Ca": {},
            "Cb": {},
            "Tr": {},
            "Tk": {},
            "F": {},
            "Q": {},
        }
        for episode in range(1, self.n_episodes + 1):
            self.env.reset()
            episode_loss = 0
            current_normed_state_tensor = torch.Tensor(
                tuple(v for v in self.env.normed_state.values())
            )
            cnt = 0
            for step in range(self.step_per_episode):
                # select action
                normed_action = self.ddpg.select_action(
                    normed_state=current_normed_state_tensor
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
                buffer_data.replay_buffer.extend(
                    TensorDict(
                        {
                            "normed_state": current_normed_state_tensor[None, :],
                            "normed_action": torch.Tensor(normed_action)[None, :],
                            "reward": torch.Tensor([step_loss])[None, :],
                            "next_normed_state": next_normed_state_tensor[None, :],
                        },
                        batch_size=[1],
                    )
                )
                current_normed_state_tensor = next_normed_state_tensor

                # Sample a random mini-batch of N transitions (si, ai, ri, si+1) from R
                sample_batch = buffer_data.replay_buffer.sample(
                    self.ddpg_kwargs.get("batch_size")
                )
                actor_loss, critic_loss = self.ddpg.update_network(sample_batch)

                self.actor_loss_history.append(actor_loss.detach().numpy().item())
                self.critic_loss_history.append(critic_loss.detach().numpy().item())

                if cnt == self.early_stop_patience:
                    break

            print(f"episode [{episode}]-------------------------------------------")
            print(f"episode loss : {round(episode_loss, ndigits=4)}")
            print(f"jitter noise : {round(self.ddpg.jitter_noise, ndigits=4)}")
            print(f"learning rate : {round(self.ddpg.learning_rate, ndigits=4)}")
            self.episode_loss_traj.append(episode_loss)
            self.ddpg.update_lr()
            if episode % 200 == 0:
                # turn to inference mode
                self.ddpg.inference = True
                (
                    Ca_traj,
                    Cb_traj,
                    Tr_traj,
                    Tk_traj,
                    F_traj,
                    Q_traj,
                ) = self.inference_once()

                inference_traj["Ca"][episode] = Ca_traj
                inference_traj["Cb"][episode] = Cb_traj
                inference_traj["Tr"][episode] = Tr_traj
                inference_traj["Tk"][episode] = Tk_traj
                inference_traj["F"][episode] = F_traj
                inference_traj["Q"][episode] = Q_traj

                # return back to training mode
                self.ddpg.inference = False

                buffer_data.save_replay_buffer()

        plot_inference_result(inference_traj=inference_traj)
        plot_reward_trend(rewards=self.episode_loss_traj)

        if save_network:
            self.ddpg.save_networks()
