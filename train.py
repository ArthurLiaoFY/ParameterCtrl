import copy
from collections import defaultdict

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

from config import env_kwargs, experiment_step, q_learning_kwargs
from cstr_env import CSTREnv, np
from q_agent import Agent

env = CSTREnv(**env_kwargs)
agent = Agent(**q_learning_kwargs)

n_episodes = 10000

rewards = []

max_total_reward = -np.inf


for episode in range(n_episodes):
    env.reset()
    total_reward = 0

    for _ in range(experiment_step):
        state = env.state.copy()
        action_idx = agent.select_action_idx(
            state_tuple=tuple(v for v in state.values())
        )
        action = agent.action_idx_to_action(action_idx=action_idx)
        reward = env.step(action=action)

        agent.update_policy(
            state_tuple=tuple(v for v in state.values()),
            action_idx=action_idx,
            reward=reward,
            next_state_tuple=tuple(v for v in env.state.values()),
        )

        total_reward += reward
    agent.update_lr_er(episode=episode)
    rewards.append(total_reward)
    if total_reward > max_total_reward:
        # print
        max_total_reward = total_reward
        print(f"Episode {episode}/{n_episodes}: Total reward : {total_reward}")


agent.save_table(prefix="test_")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(rewards)), y=rewards, mode="lines+markers"))
plotly.offline.plot(figure_or_data=fig, filename="reward_trend.html")
