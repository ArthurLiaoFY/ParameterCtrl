import plotly
import plotly.graph_objects as go

from agent import Agent
from config import training_kwargs
from cstr_env import CSTREnv, np
from plot_fns import plot_validation_result


class ValidateCSTRAgent:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

        self.reset()

    def reset(self):
        self.env = CSTREnv(seed=1122, **self.env_kwargs)
        self.agent = Agent(**self.q_learning_kwargs)

        self.max_total_reward = -np.inf
        self.rewards = []

    def valid_agent(self):
        """
        validate agent on env
        """
        valid_reward_trend = [None]
        valid_Ca_trend = [self.env_kwargs.get("init_Ca")]
        valid_T_trend = [self.env_kwargs.get("init_T")]
        valid_Tc_trend = [self.env_kwargs.get("init_Tc")]

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
            reward_trend=valid_reward_trend,
            ideal_Ca=self.env_kwargs.get("ideal_Ca"),
            ideal_T=self.env_kwargs.get("ideal_T"),
            upper_Tc=self.env_kwargs.get("upper_Tc"),
            lower_Tc=self.env_kwargs.get("lower_Tc"),
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


vcstra = ValidateCSTRAgent(**training_kwargs)
vcstra.load_table(prefix="CSTR_")
vcstra.valid_agent()
