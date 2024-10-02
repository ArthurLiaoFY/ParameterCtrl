import numpy as np

from agent.actor_critic import Actor, ActorCritic, Critic, ReplayBuffer, torch


class DDPG(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

        self.actor = Actor(
            self.ddpg_kwargs.get("state_dim"),
            self.ddpg_kwargs.get("action_dim"),
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.ddpg_kwargs.get("learning_rate"),
        )

        self.actor_prime = Actor(
            self.ddpg_kwargs.get("state_dim"),
            self.ddpg_kwargs.get("action_dim"),
        )
        self.actor_prime.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            self.ddpg_kwargs.get("state_dim"),
            self.ddpg_kwargs.get("action_dim"),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.ddpg_kwargs.get("learning_rate"),
        )

        self.critic_prime = Critic(
            self.ddpg_kwargs.get("state_dim"),
            self.ddpg_kwargs.get("action_dim"),
        )
        self.critic_prime.load_state_dict(self.critic.state_dict())

        self.max_total_reward = -np.inf

        self.rewards_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def select_action(self, state: torch.Tensor):
        return self.actor(state)

    def train(self, replay_buffer, batch_size=256):
        buffer = replay_buffer.sample(batch_size)

        target_Q = buffer.get("reward") + self.discount * self.critic_prime(
            buffer.get("next_state"),
            self.actor_prime(buffer.get("next_state")),
        )

        ### Get current Q estimate
        current_Q = self.critic(
            buffer.get("state"),
            self.actor_prime(buffer.get("state")),
        )

        ### Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)

        ### Append the critic loss to the critic_loss_list
        self.critic_loss_history.append(critic_loss)

        ### Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ### Compute actor loss
        actor_loss = -self.critic(
            buffer.get("state"),
            self.actor(buffer.get("state")),
        ).mean()

        ### Append the actor loss to the actor_loss_list.
        self.actor_loss_history.append(actor_loss)

        ### Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### Update the frozen target models
        for param, target_param in zip(
            self.critic.parameters(), self.critic_prime.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_prime.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, dir, ep):

        torch.save(self.critic.state_dict(), dir + "/model/_critic" + str(ep))
        torch.save(self.actor.state_dict(), dir + "/model/_actor" + str(ep))

    def load(self, dir, ep):

        self.critic.load_state_dict(torch.load(dir + "/model/_critic" + str(ep)))
        self.critic_prime.load_state_dict(torch.load(dir + "/model/_critic" + str(ep)))

        self.actor.load_state_dict(torch.load(dir + "/model/_actor" + str(ep)))
        self.actor_prime.load_state_dict(torch.load(dir + "/model/_actor" + str(ep)))
