
import copy
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=(256,256)):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], dim=-1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

    def Q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return torch.squeeze(q1, dim=-1)

# Define actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256,256)):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))

        return a

class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256,256),
                 batch_size=256, gamma=0.99, tau=0.005, lr=(3e-4,3e-4),
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cpu"):

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1])

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = device
        self.batch_size = batch_size

        self.critic_loss_history = []
        self.actor_loss_history = []

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, iterations=2):

        for it in range(iterations):
            self.total_it += 1
            minibatch = random.sample(replay_buffer, self.batch_size)
            state = torch.Tensor(tuple(d[0] for d in minibatch)).to(self.device)
            action = torch.Tensor(tuple(d[1] for d in minibatch)).to(self.device)
            reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
            next_state = torch.Tensor(tuple(d[3] for d in minibatch)).to(self.device)
            done = torch.Tensor(tuple(d[4] for d in minibatch)).to(self.device)

            # Critic #
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                targetQ1, targetQ2 = self.critic_target(next_state, next_action)
                targetQ = reward + (1 - done) * self.gamma * torch.min(targetQ1, targetQ2)

            currentQ1, currentQ2 = self.critic(state, action)

            critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(state)
                Q = self.critic.Q1(state, policy_actions)

                actor_loss = -Q.mean()
                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)