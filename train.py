import plotly
import plotly.graph_objects as go

from agent import Agent
from config import training_kwargs
from cstr_env import CSTREnv, np
from plot_fns import plot_validation_result


class TrainCSTRAgent:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

        self.reset()

    def reset(self):
        self.env = CSTREnv(**self.env_kwargs)
        self.agent = Agent(**self.q_learning_kwargs)

        self.max_total_reward = -np.inf
        self.rewards = []

    def train_agent(self):
        """
        train agent on env
        """
        for episode in range(self.n_episodes):
            self.env.reset()
            self.agent.shutdown_explore
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

    def valid_agent(self):
        """
        validate agent on env
        """
        valid_reward_trend = []
        valid_Ca_trend = []
        valid_T_trend = []
        valid_Tc_trend = []

        self.env.reset()
        self.agent.shutdown_explore
        for _ in range(self.step_per_episode):
            state = self.env.state.copy()
            action_idx = self.agent.select_action_idx(
                state_tuple=tuple(v for v in state.values())
            )
            action = self.agent.action_idx_to_action(action_idx=action_idx)
            reward, new_Ca, new_T, new_Tc = self.env.step(action=action, return_xy=True)
            valid_reward_trend.append(reward)
            valid_Ca_trend.append(new_Ca)
            valid_T_trend.append(new_T)
            valid_Tc_trend.append(new_Tc)

        plot_validation_result(
            Ca_trend=valid_Ca_trend,
            T_trend=valid_T_trend,
            Tc_trend=valid_Tc_trend,
            ideal_Ca=self.env_kwargs.get("ideal_Ca"),
            ideal_T=self.env_kwargs.get("ideal_T"),
            upper_Tc=self.env_kwargs.get("upper_Tc"),
            lower_Tc=self.env_kwargs.get("lower_Tc"),
        )

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

    def load_table(
        self,
        file_path: str = ".",
        prefix: str = "",
        suffix: str = "",
        table_name: str = "q_table",
    ):
        self.agent.q_table = np.load(
            f"{file_path}/{prefix}{table_name}{suffix}.npy", allow_pickle=True
        ).item()

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

tcstra.train_agent()
tcstra.valid_agent()
