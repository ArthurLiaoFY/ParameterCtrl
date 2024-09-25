import plotly
import plotly.graph_objects as go

from agent.q_table import Agent
from config import training_kwargs
from cstr_env import CSTREnv, np


class TrainCSTRAgent:
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


tcstra = TrainCSTRAgent(**training_kwargs)

tcstra.train_agent(plot_reward_trend=True)
tcstra.save_table(prefix="CSTR_")
