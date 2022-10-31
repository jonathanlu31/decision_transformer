import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import os

class TrajectoryDataset(Dataset):
    def __init__(self, base_path, context_len, state_dim, action_dim):
        self.base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), base_path)
        self.c_len = context_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.state_mean, self.state_std = self._compute_mean_std()
        self.trajectories = []
        for i in range(len(os.listdir(self.base_path))):
            interval = i * 150
            path = os.path.join(self.base_path, f'traj_{interval}-{interval + 150}.pkl')
            with open(path, 'rb') as f:
                file_trajectories = pickle.load(f)
            self.trajectories.extend(file_trajectories)
            print(i)
        print('done')

    def __len__(self):
        return len(os.listdir(self.base_path)) * 150

    def __getitem__(self, idx):
        # interval = (idx // 150) * 150
        # path = os.path.join(self.base_path, f'traj_{interval}-{interval + 150}.pkl')
        # with open(path, 'rb') as f:
        #     file_trajectories = pickle.load(f)
        # traj = file_trajectories[idx - interval]
        traj = self.trajectories[idx]

        mask = 1 - np.asarray(traj["dones"])
        returns = self._compute_returns(0, traj["rewards"], mask, gamma=1)

        rand_start = random.randint(0, len(traj['states']) - 1)
        s = np.stack(traj['states'][rand_start:rand_start+self.c_len]).reshape((-1, self.state_dim))
        a = np.stack(traj['actions'][rand_start:rand_start+self.c_len]).reshape((-1, self.action_dim))
        r = np.stack(returns[rand_start:rand_start+self.c_len]).reshape((-1, 1))
        d = np.stack(traj['dones'][rand_start:rand_start+self.c_len])
        timesteps = np.arange(rand_start, rand_start+len(s)) # TODO: check if padding cutoff is necessary and if rtg needs padding

        tlen = s.shape[0]
        s = np.concatenate((np.zeros((self.c_len - tlen, self.state_dim)), s))
        s = torch.tensor(s)
        # s = (s - self.state_mean) / self.state_std
        a = np.concatenate((np.ones((self.c_len - tlen, self.action_dim)) * -10., a))
        r = np.concatenate((np.zeros((self.c_len - tlen, 1)), r))
        d = np.concatenate((np.ones(self.c_len - tlen) * 2, d))
        timesteps = np.concatenate((np.zeros(self.c_len - tlen), timesteps))
        mask = np.concatenate((np.zeros(self.c_len - tlen), np.ones(tlen)))

        return s, a, r, d, timesteps, mask

    def _compute_returns(self, final_value, rewards, masks, gamma=0.99):
        total_reward = final_value
        discounted_rewards = [0 for _ in range(len(rewards))]
        for step in reversed(range(len(rewards))):
            total_reward = gamma * total_reward * masks[step] + rewards[step]
            discounted_rewards[step] = total_reward
        return discounted_rewards
