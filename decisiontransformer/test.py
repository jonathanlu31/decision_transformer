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
# model = DecisionTransformer(state_dim, action_dim, config["hidden_size"], c_len, 200, True, n_head=1, n_layer=3, n_inner=4*config['hidden_size'],
#         activation_function=config['activation_function'],
#         n_positions=1024,
#         resid_pdrop=config['dropout'],
#         attn_pdrop=config['dropout'], device=config["device"]).cuda()

train_dataset = TrajectoryDataset(c_len, state_dim, action_dim)
states, actions, returns, dones, timesteps, attn_mask = train_dataset[3]
# states = states.to(dtype=torch.float).unsqueeze(0).cuda()
# actions = torch.from_numpy(actions).to(dtype=torch.float).unsqueeze(0).cuda()
# returns = torch.from_numpy(returns).to(dtype=torch.float).unsqueeze(0).cuda()
# timesteps = torch.from_numpy(timesteps).to(dtype=torch.long).unsqueeze(0).cuda()
# attn_mask = torch.from_numpy(attn_mask).unsqueeze(0).cuda()
# stacked_attn_mask = torch.stack(
#             (attn_mask, attn_mask, attn_mask), dim=1
#         ).permute(0, 2, 1).reshape(attn_mask.shape[0], 1, 3*attn_mask.shape[1])
# attention_mask = stacked_attn_mask.transpose(-1, -2) @ stacked_attn_mask
# stuff = -1e9 * attention_mask
# print(attn_mask)
# action_preds = model(
#                     states, actions, returns, timesteps=timesteps, attn_mask=attn_mask
#                 )
# print(len(train_dataset))
# torch.save(model.state_dict(), 'models/mingpt')

# attn_mask = torch.cat([torch.ones(1), torch.zeros(5-1)])
# attn_mask2 = torch.cat([torch.ones(2), torch.zeros(5-2)])
# attn_mask = torch.stack([attn_mask, attn_mask2], dim=0)
# attn_mask = attn_mask.to(dtype=torch.long)
# stacked_attn_mask = torch.stack(
#             (attn_mask, attn_mask, attn_mask), dim=1
#         ).permute(0, 2, 1).reshape(2, 1, 3*attn_mask.shape[1])
# attention_mask = stacked_attn_mask.transpose(-1, -2) @ stacked_attn_mask

# print(stacked_attn_mask)
# print(attention_mask)
