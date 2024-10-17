import os

import numpy as np
import torch
from tensordict import TensorDict

from agent.double_dqn import DoubleDeepQNetwork
from cstr_env import CSTREnv
from train.collect_buffer_data import CollectBufferData

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = "dummy"


class TrainDDQN:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)
        self.ddqn_agent = DoubleDeepQNetwork(**self.ddqn_kwargs)

        self.max_train_reward = -np.inf
        self.episode_reward_traj = []
        self.dqn_loss_history = []

        self.inference_traj = {
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

    def inference_once(self, episode: int):
        # set up env
        self.env.reset()
        inference_reward = 0
        cnt = 0

        # shutdown explore
        self.ddqn_agent.shutdown_explore

        # play game
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

        self.inference_traj["Ca"][episode] = self.env.Ca_traj
        self.inference_traj["Cb"][episode] = self.env.Cb_traj
        self.inference_traj["Tr"][episode] = self.env.Tr_traj
        self.inference_traj["Tk"][episode] = self.env.Tk_traj
        self.inference_traj["F"][episode] = self.env.F_traj
        self.inference_traj["Q"][episode] = self.env.Q_traj

        print(f"[{episode:06d}] inference reward: {inference_reward:.4f}")

        # restart explore
        self.ddqn_agent.start_explore

    def train_agent(
        self,
        buffer_data: CollectBufferData,
        save_traj_to_buffer: bool = True,
        save_network: bool = True,
    ):
        for episode in range(self.ddqn_kwargs.get("n_episodes") + 1):
            if episode % self.inference_per_episode == 0:
                self.inference_once(episode=episode, save_animate=False)
            else:
                # set up env
                train_reward = 0
                self.env.reset_game()

                state_list = []
                action_idx_list = []
                reward_list = []
                next_state_list = []

                if episode % self.ddqn_kwargs.get("update_target_each_k_episode") == 0:
                    self.ddqn_agent.dqn_prime.load_state_dict(
                        self.ddqn_agent.dqn.state_dict()
                    )

                while not self.env.game_over():
                    # state
                    state_tuple = scale_state_to_tuple(
                        state_dict=self.env.getGameState(),
                        state_scale=self.feature_scaling,
                    )
                    state_list.append(state_tuple)

                    # action
                    action_idx = self.ddqn_agent.select_action_idx(state_tuple)
                    action_idx_list.append(action_idx)

                    # reward
                    reward = self.env.act(self.env.getActionSet()[action_idx])
                    next_state_dict = self.env.getGameState()
                    redefined_reward = reward_redefine(
                        state_dict=next_state_dict,
                        reward=reward,
                    )
                    reward_list.append(redefined_reward)

                    # next state
                    next_state_tuple = scale_state_to_tuple(
                        state_dict=next_state_dict,
                        state_scale=self.feature_scaling,
                    )
                    next_state_list.append(next_state_tuple)

                    # cumulate reward
                    train_reward += redefined_reward

                    sample_batch = buffer_data.replay_buffer.sample(
                        batch_size=self.ddqn_kwargs.get("batch_size")
                    )
                    # update agent policy per step
                    self.ddqn_agent.update_policy(
                        episode=episode,
                        sample_batch=sample_batch,
                    )

                if len(state_list) > 0:
                    buffer_data.replay_buffer.extend(
                        TensorDict(
                            {
                                "state": torch.Tensor(np.array(state_list)),
                                "action_idx": torch.Tensor(np.array(action_idx_list))[
                                    :, None
                                ],
                                "reward": torch.Tensor(np.array(reward_list))[:, None],
                                "next_state": torch.Tensor(np.array(next_state_list)),
                            },
                            batch_size=[len(state_list)],
                        )
                    )

                # update status
                self.ddqn_agent.update_lr_er(episode=episode)
                if train_reward > self.max_train_reward:
                    print(
                        f"[{episode:06d}] max_train_reward updated from {self.max_train_reward:.4f} to {train_reward:.4f}"
                    )
                    self.max_train_reward = train_reward

                # record reward
                self.episode_reward_traj.append(train_reward)
