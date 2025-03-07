# %%
from agent.ddpg_agent import DeepDeterministicPolicyGradient
from agent.dqn_agent import DeepQNetwork
from config import training_kwargs
from env.cartpole_env import CartPole
from train.train_cartpole import TrainAgent

env = CartPole()

agent = DeepQNetwork(**training_kwargs.get("agent_kwargs"))
tddpg = TrainAgent(env=env, agent=agent, **training_kwargs)
tddpg.train_online_agent(
    save_traj_to_buffer=False,
    save_network=False,
)

# %%
