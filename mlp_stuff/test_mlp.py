import torch
import torch.nn as nn
import numpy as np
import gym
from dataset.trajectory_dataset import TrajectoryDataset

env = gym.make('CartPole-v1')
    
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

train_dataset = TrajectoryDataset(None, state_dim, action_dim)
returns = np.array([train_dataset[i] for i in train_dataset])
print(np.count_nonzero(200 - returns))
print(200 - returns)