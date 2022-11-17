import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, num_hidden):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.actor = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_size),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        dist = torch.distributions.Categorical(probs)
        return dist, value