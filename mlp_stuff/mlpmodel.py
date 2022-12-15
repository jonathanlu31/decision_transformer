import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state_input, r2g=None):
        # state_embedding = self.embed_state(state_input)
        # print(state_input)
        # print(state_embedding)
        action_preds = self.mlp(state_input)
        return action_preds
