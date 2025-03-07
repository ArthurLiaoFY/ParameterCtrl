# %%
from agent.ddpg_agent import DeepDeterministicPolicyGradient
from config import training_kwargs
from env.cstr_env import CSTREnv
from train.train_cstr import TrainCSTR

env = CSTREnv(**training_kwargs.get("cstr_env_kwargs"))
agent = DeepDeterministicPolicyGradient(
    state_dim=training_kwargs.get("cstr_env_kwargs").get("state_dim"),
    action_dim=training_kwargs.get("cstr_env_kwargs").get("action_dim"),
    **training_kwargs.get("agent_kwargs")
)
# %%

tddpg = TrainCSTR(env=env, agent=agent, **training_kwargs)
tddpg.train_online_agent(
    save_traj_to_buffer=False,
    save_network=False,
)
# %%
