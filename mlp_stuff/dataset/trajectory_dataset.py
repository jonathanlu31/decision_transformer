import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import os

class TrajectoryDataset(Dataset):
    def __init__(self, context_len, state_dim, action_dim):
        self.c_len = context_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trajectories = []
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'traj_dataset_good_large.pkl')
        with open(path, 'rb') as f:
            self.trajectories = pickle.load(f)
        print('done')
        self.state_mean, self.state_std = self._compute_mean_std()

    def __len__(self):
        return len(self.trajectories) * 200

    def __getitem__(self, idx):
        traj_idx = idx // 200
        traj = self.trajectories[traj_idx]
        timestep_idx = idx - traj_idx * 200

        mask = 1 - np.asarray(traj["dones"])
        returns = self._compute_returns(0, traj["rewards"], mask, gamma=1)
        return traj['states'][timestep_idx].cpu(), traj['actions'][timestep_idx], returns[timestep_idx]

    def _compute_returns(self, final_value, rewards, masks, gamma=0.99):
        total_reward = final_value
        discounted_rewards = [0 for _ in range(len(rewards))]
        for step in reversed(range(len(rewards))):
            total_reward = gamma * total_reward * masks[step] + rewards[step]
            discounted_rewards[step] = total_reward
        return discounted_rewards

    def _compute_mean_std(self):
        states = torch.cat([torch.concatenate(traj['states'], axis=0) for traj in self.trajectories], 0)
        return torch.mean(states, axis=0).cpu(), torch.std(states, axis=0).cpu() + 1e-6
