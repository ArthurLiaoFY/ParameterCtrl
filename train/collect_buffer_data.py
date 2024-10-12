from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from tqdm import tqdm

from agent.actor_critic import ReplayBuffer, torch
from cstr_env import CSTREnv, np


class CollectBufferData:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
        self.env = CSTREnv(**self.env_kwargs)

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_kwargs.get("buffer_size"),
            )
        )
        self.load_replay_buffer()

    def extend_buffer_data(self, extend_amount: int, save: bool = True) -> None:
        for _ in tqdm(range(extend_amount)):
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
                        "normed_state": torch.Tensor(np.array(normed_state_list)),
                        "normed_action": torch.Tensor(np.array(normed_action_list)),
                        "reward": torch.Tensor(np.array(reward_list)),
                        "next_normed_state": torch.Tensor(
                            np.array(next_normed_state_list)
                        ),
                    },
                    batch_size=[self.step_per_episode],
                )
            )

        if save:
            self.save_replay_buffer()

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
