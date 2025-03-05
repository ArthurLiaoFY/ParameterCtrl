# %%
from config import training_kwargs
from cstr_env import CSTREnv, np
from train.collect_buffer_data import CollectBufferData
from train.train_ddpg import TrainAgent

env = CSTREnv(seed=1122, **training_kwargs.get("env_kwargs"))
buffer_data = CollectBufferData(env=env, **training_kwargs)
buffer_data.init_buffer_data(init_amount=10)
# %%
a = buffer_data.sample_buffer_data(size=20000)
# %%
tddpg = TrainAgent(**training_kwargs)
tddpg.train_online_agent(
    buffer_data=buffer_data,
    save_traj_to_buffer=False,
    save_network=True,
)
