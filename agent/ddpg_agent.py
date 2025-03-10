import numpy as np
import torch
from tensordict import TensorDict

from agent.agent import RLAgent
from agent.model.actor import Actor
from agent.model.critic import Critic


class DeepDeterministicPolicyGradient(RLAgent):
    def __init__(self, state_dim, action_dim, **kwargs) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.__dict__.update(**kwargs)
        self.start_explore

        self.actor = Actor(
            self.state_dim,
            self.action_dim,
        )
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
        )
        self.load_network()

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

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )

    def select_action(self, state: torch.Tensor):
        if not self.explore:
            additional_noise = np.array([0.0 for _ in range(self.action_dim)])
        else:
            self.jitter_noise = max(
                self.jitter_noise_min,
                self.jitter_noise * self.jitter_noise_decay_factor,
            )
            additional_noise = np.random.randn() * self.jitter_noise

        with torch.no_grad():
            action = self.actor(state).detach().numpy() + additional_noise

        return action

    def update_policy(self, sample_batch: TensorDict) -> None:
        # Set yi(next_action_score) = ri + γ * Q_prime(si + 1, µ_prime(si + 1 | θ ^ µ_prime) | θ ^ Q_prime)
        with torch.no_grad():
            td_target = sample_batch.get(
                "reward"
            ) + self.discount_factor * self.critic_prime(
                sample_batch.get("next_state"),
                self.actor_prime(sample_batch.get("next_state")),
            )

        # Update critic by minimizing the mse loss
        current_reward = self.critic(
            sample_batch.get("state"),
            sample_batch.get("action"),
        )

        critic_loss = torch.nn.functional.huber_loss(current_reward, td_target)

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

        # θ′ ← τ θ + (1 − τ) θ′
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

        return None

    def update_lr(self) -> None:
        self.learning_rate = max(
            self.learning_rate_min,
            self.learning_rate * self.learning_rate_decay_factor,
        )

    @property
    def shutdown_explore(self) -> None:
        self.explore = False

    @property
    def start_explore(self) -> None:
        self.explore = True

    def save_network(
        self,
        model_file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        actor_name: str = "ddpg_actor_network",
        critic_name: str = "ddpg_critic_network",
    ):
        torch.save(
            self.actor.state_dict(),
            f"{model_file_path}/{prefix}{actor_name}{suffix}.pt",
        )
        torch.save(
            self.critic.state_dict(),
            f"{model_file_path}/{prefix}{critic_name}{suffix}.pt",
        )

    def load_network(
        self,
        model_file_path: str = "./agent/trained_agent",
        prefix: str = "",
        suffix: str = "",
        actor_name: str = "ddpg_actor_network",
        critic_name: str = "ddpg_critic_network",
    ):
        try:
            self.actor.load_state_dict(
                torch.load(
                    f"{model_file_path}/{prefix}{actor_name}{suffix}.pt",
                    weights_only=True,
                )
            )
            self.critic.load_state_dict(
                torch.load(
                    f"{model_file_path}/{prefix}{critic_name}{suffix}.pt",
                    weights_only=True,
                )
            )
            print(
                f"Found trained model under {model_file_path}, weights have been loaded."
            )
        except FileNotFoundError:
            pass
