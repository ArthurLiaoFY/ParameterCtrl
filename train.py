# %%
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

    def extend_buffer_data(self, save: bool = True) -> None:
        for _ in tqdm(range(self.n_episodes)):
            self.env.reset()
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []

            for _ in range(self.step_per_episode):
                state_list.append(tuple(v for v in self.env.state.values()))
                action = (
                    np.random.uniform(low=5.0, high=100.0, size=1).item(),
                    np.random.uniform(low=-8500, high=0.0, size=1).item(),
                )
                action_list.append(action)
                reward_list.append(self.env.step(action=action))
                next_state_list.append(tuple(v for v in self.env.state.values()))

            self.replay_buffer.extend(
                TensorDict(
                    {
                        "state": torch.Tensor(state_list),
                        "action": torch.Tensor(action_list),
                        "reward": torch.Tensor(reward_list),
                        "next_state": torch.Tensor(next_state_list),
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
        self.ddpg = DDPG(**self.ddpg_kwargs)

    def train_agent(self, replay_buffer: ReplayBuffer, plot_reward_trend: bool = False):
        for episode in range(self.n_episodes):
            self.env.reset()
            total_reward = 0

            state = self.env.state.copy()
            action = self.ddpg.actor(tuple(v for v in state.values()))
            reward = self.env.step(action=action)

            self.ddpg.train(
                replay_buffer=replay_buffer,
            )
        if plot_reward_trend:
            self.plot_reward_trend()

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


# tcstra = TrainQAgent(**training_kwargs)

# tcstra.train_agent(plot_reward_trend=True)
# tcstra.save_table(prefix="CSTR_Q_")

# %%
cbd = CollectBufferData(**training_kwargs)
# cbd.extend_buffer_data()
print(cbd.replay_buffer)
# %%
a = cbd.replay_buffer.sample(1)
# %%
print(
    a["action"],
    a["next_state"],
    a["state"],
    a["reward"],
)


# %%
