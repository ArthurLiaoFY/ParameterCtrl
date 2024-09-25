import plotly
import plotly.graph_objects as go

from agent import Agent
from config import training_kwargs
from cstr_env import CSTREnv, np


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
        valid_Cb_trend = [self.env_kwargs.get("init_Cb")]
        valid_Tr_trend = [self.env_kwargs.get("init_Tr")]
        valid_Tk_trend = [self.env_kwargs.get("init_Tk")]

        self.env.reset()
        self.agent.shutdown_explore
        for _ in range(self.step_per_episode):
            state = self.env.state.copy()
            action_idx = self.agent.select_action_idx(
                state_tuple=tuple(v for v in state.values())
            )
            action = self.agent.action_idx_to_action(action_idx=action_idx)
            reward, new_Ca, new_Cb, new_Tr, new_Tk, new_F, new_Q = self.env.step(
                action=action, return_xy=True
            )
            valid_reward_trend.append(reward)
            valid_Ca_trend.append(new_Ca)
            valid_Cb_trend.append(new_Cb)
            valid_Tr_trend.append(new_Tr)
            valid_Tk_trend.append(new_Tk)

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
