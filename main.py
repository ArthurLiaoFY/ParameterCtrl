# %%
from agent.ddpg_agent import DeepDeterministicPolicyGradient
from agent.dqn_agent import DeepQNetwork
from config import training_kwargs
from env.cstr_env import CSTREnv
from train.train_cstr import TrainAgent

env = CSTREnv(**training_kwargs.get("env_kwargs"))
# agent = DeepDeterministicPolicyGradient(**training_kwargs.get("agent_kwargs"))
agent = DeepQNetwork(**training_kwargs.get("agent_kwargs"))
# %%

tddpg = TrainAgent(env=env, agent=agent, **training_kwargs)
tddpg.train_online_agent(
    save_traj_to_buffer=False,
    save_network=False,
)
# %%