import torch
import torch.nn as nn
import numpy as np
import gym
from decision_transformer import DecisionTransformer
from dataset.trajectory_dataset import TrajectoryDataset

env = gym.make('CartPole-v1')
config = {
        "learning_rate": 2e-4,
        "epochs": 100,
        "batch_size": 32,
        "hidden_size": 64,
        "c_len": 20,
        "device": "auto",
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "activation_function": "relu",
        'dropout': 0.1,
        "warmup_steps": 10000,
        "num_workers": 0
    }
    
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
c_len = config["c_len"]
model = DecisionTransformer(state_dim, action_dim, config["hidden_size"], c_len, 200, True, n_head=1, n_layer=3, n_inner=4*config['hidden_size'],
        activation_function=config['activation_function'],
        n_positions=1024,
        resid_pdrop=config['dropout'],
        attn_pdrop=config['dropout'], device=config["device"])

# train_dataset = TrajectoryDataset(c_len, state_dim, action_dim)
# print(len(train_dataset))
torch.save(model.state_dict(), 'models/mingpt')
