# %%
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from tqdm import tqdm

from agent.actor_critic import Actor, ActorCritic, Critic, ReplayBuffer, torch
from agent.ddpg import DDPG
from agent.q_agent import Agent
from config import training_kwargs
from cstr_env import CSTREnv, np


class CollectBufferData:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.ddpg_kwargs.get("buffer_size"),
            )
        )

        try:
            self.load_replay_buffer()

        except FileNotFoundError:
            pass

    def extend_buffer_data(self, extend_amount: int, save: bool = True) -> None:
        for _ in tqdm(range(extend_amount)):
            self.env.reset()
            normed_state_list = []
            normed_action_list = []
            reward_list = []
            next_normed_state_list = []

            for _ in range(self.step_per_episode):
                normed_state_list.append(
                    tuple(v for v in self.env.normed_state.values())
                )
                action = np.array(
                    [
                        np.random.uniform(
                            low=self.env_kwargs.get("lower_F"),
                            high=self.env_kwargs.get("upper_F"),
                            size=1,
                        ).item()
                        - self.env.state.get("current_F"),
                        np.random.uniform(
                            low=self.env_kwargs.get("lower_Q"),
                            high=self.env_kwargs.get("upper_Q"),
                            size=1,
                        ).item()
                        - self.env.state.get("current_Q"),
                    ]
                )

                normed_action_list.append(self.env.norm_action(action))
                reward_list.append(self.env.step(action=action))
                next_normed_state_list.append(
                    tuple(v for v in self.env.normed_state.values())
                )

            self.replay_buffer.extend(
                TensorDict(
                    {
                        "normed_state": torch.Tensor(np.array(normed_state_list)),
                        "normed_action": torch.Tensor(np.array(normed_action_list)),
                        "reward": torch.Tensor(np.array(reward_list)),
                        "next_normed_state": torch.Tensor(
                            np.array(next_normed_state_list)
                        ),
                    },
                    batch_size=[self.step_per_episode],
                )
            )

        if save:
            self.save_replay_buffer()

    def save_replay_buffer(self) -> None:
        self.replay_buffer.dumps(self.ddpg_kwargs.get("replay_buffer_dir"))
        pass

    def load_replay_buffer(self) -> None:
        self.replay_buffer.loads(self.ddpg_kwargs.get("replay_buffer_dir"))
        pass


class TrainQAgent:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

        self.reset()

    def reset(self):
        self.env = CSTREnv(**self.env_kwargs)
        self.agent = Agent(**self.q_learning_kwargs)

        self.max_total_reward = -np.inf
        self.rewards = []

    def train_agent(self, plot_reward_trend: bool = False):
        """
        train agent on env
        """
        for episode in range(self.n_episodes):
            self.env.reset()
            total_reward = 0

            for _ in range(self.step_per_episode):
                state = self.env.state.copy()
                action_idx = self.agent.select_action_idx(
                    state_tuple=tuple(v for v in state.values())
                )
                action = self.agent.action_idx_to_action(action_idx=action_idx)
                reward = self.env.step(action=action)

                self.agent.update_policy(
                    state_tuple=tuple(v for v in state.values()),
                    action_idx=action_idx,
                    reward=reward,
                    next_state_tuple=tuple(v for v in self.env.state.values()),
                )

                total_reward += reward
            self.agent.update_lr_er(episode=episode)
            self.rewards.append(total_reward)
            if total_reward > self.max_total_reward:
                # print
                self.max_total_reward = total_reward
                print(
                    f"Episode {episode}/{self.n_episodes}: Total reward : {total_reward}"
                )
        if plot_reward_trend:
            self.plot_reward_trend()

    def save_table(
        self,
        file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        table_name: str = "q_table",
    ):
        self.agent.save_table(
            file_path=file_path, prefix=prefix, suffix=suffix, table_name=table_name
        )

    def plot_reward_trend(
        self,
        file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        fig_name: str = "reward_trend",
    ):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.rewards)), y=self.rewards, mode="lines+markers"
            )
        )
        plotly.offline.plot(
            figure_or_data=fig, filename=f"{file_path}/{prefix}{fig_name}{suffix}.html"
        )


class TrainDDPG:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)
        self.ddpg = DDPG(**self.ddpg_kwargs)

        self.max_total_reward = -np.inf

        self.episode_loss_traj = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def inference_once(
        self,
        episode: int,
        file_path: str = "./plots",
    ):
        self.env.reset()
        inference_reward = 0

        for step in range(self.step_per_episode):
            normed_action = self.ddpg.select_action(
                normed_state=torch.Tensor(
                    tuple(v for v in self.env.normed_state.values())
                )
            )
            inference_reward += self.env.step(
                action=self.env.revert_normed_action(normed_action=normed_action)
            )

        print(f"inference_reward: {inference_reward}")
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)
        axs[0, 0].plot(self.env.Ca_traj, "o-")
        axs[0, 0].plot(
            [self.env_kwargs.get("ideal_Ca") for _ in range(len(self.env.Ca_traj))]
        )
        axs[0, 0].set_title("Ca")

        axs[0, 1].plot(self.env.Cb_traj, "o-")
        axs[0, 1].plot(
            [self.env_kwargs.get("ideal_Cb") for _ in range(len(self.env.Cb_traj))]
        )
        axs[0, 1].set_title("Cb")

        axs[1, 0].plot(self.env.Tr_traj, "o-")
        axs[1, 0].plot(
            [self.env_kwargs.get("ideal_Tr") for _ in range(len(self.env.Tr_traj))]
        )
        axs[1, 0].set_title("Tr")

        axs[1, 1].plot(self.env.Tk_traj, "o-")
        axs[1, 1].plot(
            [self.env_kwargs.get("ideal_Tk") for _ in range(len(self.env.Tk_traj))]
        )
        axs[1, 1].set_title("Tk")
        plt.savefig(f"{file_path}/observed_value_trend_in_{episode}.png", dpi=150)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 7), sharex=True)
        ax1.plot(self.env.F_traj, "o-")
        ax1.set_title("F")

        ax2.plot(self.env.Q_traj, "o-")
        ax2.set_title("Q dot")
        plt.savefig(f"{file_path}/input_value_trend_in_{episode}.png", dpi=150)

    def train_agent(
        self,
        replay_buffer: ReplayBuffer,
        plot_loss_trend: bool = False,
    ):
        for episode in range(1, self.n_episodes + 1):
            self.env.reset()
            episode_loss = 0
            current_normed_state_tensor = torch.Tensor(
                tuple(v for v in self.env.normed_state.values())
            )
            for step in range(self.step_per_episode):
                # select action

                normed_action = self.ddpg.select_action(
                    normed_state=current_normed_state_tensor
                )
                # TODO: torch.Tensor(tuple(v for v in self.env.state.values())) to function
                step_loss = self.env.step(
                    action=self.env.revert_normed_action(normed_action=normed_action)
                )
                episode_loss += step_loss

                next_normed_state_tensor = torch.Tensor(
                    tuple(v for v in self.env.normed_state.values())
                )
                replay_buffer.extend(
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
                sample_batch = replay_buffer.sample(self.ddpg_kwargs.get("batch_size"))
                actor_loss, critic_loss = self.ddpg.update_network(sample_batch)

                self.actor_loss_history.append(actor_loss.detach().numpy().item())
                self.critic_loss_history.append(critic_loss.detach().numpy().item())

            print("-------------------------------------------")
            print(f"episode loss [{episode}] : {round(episode_loss, ndigits=4)}")
            print(
                f"jitter noise [{episode}] : {round(self.ddpg.jitter_noise, ndigits=4)}"
            )
            self.episode_loss_traj.append(episode_loss)
            self.ddpg.update_lr()
            if episode % 50 == 0:
                # turn to inference mode
                self.ddpg.inference = True
                self.inference_once(episode=episode)
                # return back to training mode
                self.ddpg.inference = False

        if plot_loss_trend:
            self.plot_loss_trend()

    # def update_buffer_data(self):
    #     replay_buffer.dumps(self.ddpg_kwargs.get("replay_buffer_dir"))

    def plot_loss_trend(
        self,
        file_path: str = "./plots",
        prefix: str = "",
        suffix: str = "",
        fig_name: str = "reward_trend",
    ):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.episode_loss_traj)),
                y=self.episode_loss_traj,
                mode="lines+markers",
            )
        )
        plotly.offline.plot(
            figure_or_data=fig, filename=f"{file_path}/{prefix}{fig_name}{suffix}.html"
        )


# tcstra = TrainQAgent(**training_kwargs)

# tcstra.train_agent(plot_reward_trend=True)
# tcstra.save_table(prefix="CSTR_Q_")
