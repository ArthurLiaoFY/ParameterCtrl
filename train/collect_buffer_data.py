import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tqdm import tqdm

from cstr_env import CSTREnv, np


class CollectBufferData:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)

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

    def init_buffer_data(self, init_amount: int) -> None:
        for _ in tqdm(range(init_amount)):
            self.env.reset()
            normed_state_list = []
            normed_action_list = []
            reward_list = []
            next_normed_state_list = []

            for _ in range(self.step_per_episode):
                normed_state_list.append(
                    tuple(v for v in self.env.normed_state.values())
                )
                action = np.array(
                    [
                        np.random.uniform(
                            low=self.env_kwargs.get("lower_F"),
                            high=self.env_kwargs.get("upper_F"),
                            size=1,
                        ).item()
                        - self.env.state.get("current_F"),
                        np.random.uniform(
                            low=self.env_kwargs.get("lower_Q"),
                            high=self.env_kwargs.get("upper_Q"),
                            size=1,
                        ).item()
                        - self.env.state.get("current_Q"),
                    ]
                )

                normed_action_list.append(self.env.norm_action(action))
                reward_list.append(self.env.step(action=action))
                next_normed_state_list.append(
                    tuple(v for v in self.env.normed_state.values())
                )

            self.replay_buffer.extend(
                TensorDict(
                    {
                        "state": torch.Tensor(np.array(normed_state_list)),
                        "action": torch.Tensor(np.array(normed_action_list)),
                        "reward": torch.Tensor(np.array(reward_list)),
                        "next_state": torch.Tensor(np.array(next_normed_state_list)),
                        "priority": torch.Tensor(-1 * np.array(reward_list)),
                    },
                    batch_size=[self.step_per_episode],
                )
            )
            self.replay_buffer.update_priority(
                index=torch.tensor([i for i in range(self.step_per_episode)]),
                priority=torch.Tensor(-1 * np.array(reward_list)),
            )

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
                    "priority": torch.Tensor(-1 * np.array(reward)),
                },
                batch_size=[len(reward)],
            )
        )
        self.replay_buffer.update_priority(
            index=torch.tensor(
                [i for i in range(self.replay_buffer.__len__(), len(reward))]
            ),
            priority=torch.Tensor(-1 * np.array(reward)),
        )

    def sample_buffer_data(self):
        pass

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
