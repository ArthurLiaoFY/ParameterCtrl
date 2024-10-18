from config import training_kwargs
from train.collect_buffer_data import CollectBufferData
from train.train_ddpg import TrainDDPG
from train.train_ddqn import TrainDDQN

buffer_data = CollectBufferData(**training_kwargs)
# buffer_data.extend_buffer_data(extend_amount=1000)

# tddpg = TrainDDPG(**training_kwargs)
# tddpg.train_agent(
#     buffer_data=buffer_data,
#     save_traj_to_buffer=False,
#     save_network=True,
# )

tddpg = TrainDDQN(**training_kwargs)
tddpg.train_agent(
    buffer_data=buffer_data,
    save_traj_to_buffer=False,
    save_network=False,
)
