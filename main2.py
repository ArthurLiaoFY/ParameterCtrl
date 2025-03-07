# %%
from agent.q_agent import QAgent

# from agent.dqn_agent import DeepQNetwork
from config import training_kwargs
from env.cartpole_env import CartPole
from train.train_cartpole import TrainCartPole

env = CartPole()

agent = QAgent(
    state_dim=training_kwargs.get("cartpole_env_kwargs").get("state_dim"),
    action_dim=training_kwargs.get("cartpole_env_kwargs").get("action_dim"),
    **training_kwargs.get("agent_kwargs")
)
tdqn = TrainCartPole(env=env, agent=agent, **training_kwargs)
tdqn.train_online_agent(
    save_traj_to_buffer=False,
    save_network=False,
)

# %%
