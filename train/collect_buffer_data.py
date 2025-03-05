import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tqdm import tqdm


class CollectBufferData:
    def __init__(self, env, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = env

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_kwargs.get("buffer_size"),
            ),
            sampler=PrioritizedSampler(
                max_capacity=int(self.replay_buffer_kwargs.get("buffer_size")),
                alpha=self.replay_buffer_kwargs.get("prioritized_sampler").get("alpha"),
                beta=self.replay_buffer_kwargs.get("prioritized_sampler").get("beta"),
            ),
        )
        self.load_replay_buffer()

    def extend_buffer_data(
        self, state: list, action: list, reward: list, next_state: list
    ) -> None:

        self.replay_buffer.extend(
            TensorDict(
                {
                    "state": torch.Tensor(np.array(state)),
                    "action": torch.Tensor(np.array(action)),
                    "reward": torch.Tensor(np.array(reward)),
                    "next_state": torch.Tensor(np.array(next_state)),
                },
                batch_size=[len(reward)],
            )
        )
        self.replay_buffer.update_priority(
            index=torch.tensor(
                [self.replay_buffer.__len__() + i for i in range(len(reward))]
            ),
            priority=torch.Tensor(-1 * np.array(reward)),
        )

    def sample_buffer_data(self, size: int):
        return self.replay_buffer.sample(batch_size=size)

    def save_replay_buffer(self) -> None:
        print(
            "buffer data save to dir: {replay_buffer_dir}".format(
                replay_buffer_dir=self.replay_buffer_kwargs.get("replay_buffer_dir")
            )
        )
        self.replay_buffer.dumps(self.replay_buffer_kwargs.get("replay_buffer_dir"))

    def load_replay_buffer(self) -> None:

        try:
            self.replay_buffer.loads(self.replay_buffer_kwargs.get("replay_buffer_dir"))
            print(
                "buffer data load from dir: {replay_buffer_dir}".format(
                    replay_buffer_dir=self.replay_buffer_kwargs.get("replay_buffer_dir")
                )
            )

        except FileNotFoundError:
            pass
