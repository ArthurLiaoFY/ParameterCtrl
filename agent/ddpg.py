import numpy as np
from tensordict import TensorDict

from agent.actor_critic import Actor, ActorCritic, Critic, ReplayBuffer, torch


class DDPG(object):
    def __init__(self, inference: bool = False, **kwargs) -> None:
        self.inference = inference
        self.__dict__.update(**kwargs)
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
        )
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
        )
        if self.inference:
            # load model
            pass
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=self.learning_rate,
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=self.learning_rate,
            )

            self.actor_prime = Actor(
                self.state_dim,
                self.action_dim,
            )
            self.actor_prime.load_state_dict(self.actor.state_dict())

            self.critic_prime = Critic(
                self.state_dim,
                self.action_dim,
            )
            self.critic_prime.load_state_dict(self.critic.state_dict())

    def select_action(self, state: torch.Tensor):
        return self.actor(state).detach().numpy()

    def update_network(self, sample_batch: TensorDict):
        # Set yi(next_action_score) = ri + γ * Q_prime(si + 1, µ_prime(si + 1 | θ ^ µ_prime) | θ ^ Q_prime)

        next_action_score = sample_batch.get("reward")[
            :, None
        ] + self.discount_factor * self.critic(
            sample_batch.get("next_state"),
            self.actor_prime(sample_batch.get("next_state")),
        )

        # Update critic by minimizing the mse loss
        current_action_score = self.critic(
            sample_batch.get("state"),
            sample_batch.get("action"),
        )

        critic_loss = torch.nn.functional.mse_loss(
            current_action_score, next_action_score
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update the actor policy using the sampled policy gradient:

        ### Compute actor loss (gradient decent so multiply -1)
        actor_loss = (
            -1
            * self.critic(
                sample_batch.get("state"),
                self.actor(sample_batch.get("state")),
            ).mean()
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks:
        with torch.no_grad():
            for critic, critic_prime in zip(
                self.critic.parameters(), self.critic_prime.parameters()
            ):
                critic_prime.data.copy_(
                    ((1 - self.tau) * critic_prime.data) + self.tau * critic.data
                )

            for actor, actor_prime in zip(
                self.actor.parameters(), self.actor_prime.parameters()
            ):
                actor_prime.data.copy_(
                    ((1 - self.tau) * actor_prime.data) + self.tau * actor.data
                )

        return (
            actor_loss,
            critic_loss,
        )

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min, self.learning_rate * self.learning_rate_decay
        )

    def save_networks(
        self,
        file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        actor_name: str = "actor_network",
        critic_name: str = "critic_network",
    ):
        pass

    def load_networks(
        self,
        file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        actor_name: str = "actor_network",
        critic_name: str = "critic_network",
    ):
        pass
