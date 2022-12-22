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
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'trajectories.pkl')
        with open(path, 'rb') as f:
            self.trajectories = pickle.load(f)
        print('done')
        self.state_mean, self.state_std = self._compute_mean_std()
        # self.save_mean_std()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        mask = 1 - np.asarray(traj["dones"])
        returns = self._compute_returns(0, traj["rewards"], mask, gamma=1)

        rand_start = random.randint(0, len(traj['states']) - self.c_len)
        seq_len = self.c_len
        if rand_start == 0:
            seq_len = random.randint(1, self.c_len)
        s = torch.stack(traj['states'][rand_start:rand_start+seq_len]).reshape((-1, self.state_dim))
        a = np.argmax(np.stack(traj['actions'][rand_start:rand_start+seq_len]).reshape((-1, self.action_dim)), axis=1)
        r = np.stack(returns[rand_start:rand_start+seq_len]).reshape((-1, 1))
        d = np.stack(traj['dones'][rand_start:rand_start+seq_len])
        timesteps = np.arange(rand_start, rand_start+len(s)) # TODO: check if padding cutoff is necessary and if rtg needs padding

        tlen = s.shape[0]
        s = s.cpu()
        s = torch.concatenate((s, torch.zeros((self.c_len - tlen, self.state_dim))))
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate((a, np.ones(self.c_len - tlen) * -10.))[:, np.newaxis]
        r = np.concatenate((r, np.zeros((self.c_len - tlen, 1))))
        d = np.concatenate((d, np.ones(self.c_len - tlen) * 2))
        timesteps = np.concatenate((timesteps, np.zeros(self.c_len - tlen)))
        mask = np.concatenate((np.ones(tlen), np.zeros(self.c_len - tlen)))

        return s, a, r, d, timesteps, mask

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

    def save_mean_std(self):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'trajectories_metadata.pkl'), 'wb') as f:
            pickle.dump((self.state_mean, self.state_std), f)
