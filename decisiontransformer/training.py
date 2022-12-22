import torch
import torch.nn as nn
import numpy as np
import time
import gym
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from decision_transformer import DecisionTransformer
from dataset.trajectory_dataset import TrajectoryDataset
import sys
import os
import pickle
import random

class Trainer:
    def __init__(self, config, model, train_dataset, loss_fn):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        if config["device"] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = config["device"]
        
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # self.optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=config["learning_rate"],
        #     weight_decay=config['weight_decay']
        # )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lambda steps: min((steps+1)/config['warmup_steps'], 1)
        # )
        # self.optimizer = self.model.configure_optimizers(config)
        self.scheduler = None
        self.loss_fn = loss_fn
        self.train_loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

    def train(self):
        model, config = self.model, self.config
        train_losses = []

        model.train()
        print('train start')
        for epoch in tqdm(range(config['epochs'])):
            batch_losses = []
            for batch_num, sample in enumerate(tqdm(self.train_loader)):
                states, actions, returns, dones, timesteps, attn_mask = sample
                states = states.to(dtype=torch.float, device=self.device)
                actions = actions.to(dtype=torch.float, device=self.device)
                returns = returns.to(dtype=torch.float, device=self.device)
                timesteps = timesteps.to(dtype=torch.long, device=self.device)
                self.optimizer.zero_grad()

                action_target = torch.clone(actions)

                action_preds = self.model(
                    states, actions, returns, timesteps=timesteps, attn_mask=attn_mask
                )

                action_preds = action_preds.reshape(-1, 1)[attn_mask.reshape(-1) > 0]
                action_target = action_target.reshape(-1, 1)[attn_mask.reshape(-1) > 0]
                loss = self.loss_fn(action_preds, action_target)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                batch_losses.append(loss.detach().cpu().item())
                wandb.log({'loss': loss.item()})
            train_losses.extend(batch_losses)
            print(np.mean(np.asarray(batch_losses)))
        
        self.train_losses = train_losses
        torch.save(model.cpu().state_dict(), 'models/gpt2_with_pad')

    @staticmethod
    def evaluate(model, env, state_mean, state_std, device, target_return=200):
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.eval()
        model.to(device=device)
        state_dim, act_dim = env.observation_space.shape[0], 1

        state_mean = state_mean.to(device=device)
        state_std = state_std.to(device=device)

        state, _info = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        ret_to_go = torch.zeros(1, device=device, dtype=torch.float32)
        ret_to_go[0] = target_return
        timesteps = torch.zeros(1, device=device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for i in range(200):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            norm_state = (states.to(dtype=torch.float32) - state_mean) / state_std

            action = model.get_action(
                norm_state,
                actions.to(dtype=torch.float32),
                ret_to_go.to(dtype=torch.float32),
                timesteps
            )
            action = action.detach().cpu()
            env_action = (1 if action > 0.5 else 0)
            print(action)
            print(env_action)
            state, reward, done, _truncated, _info = env.step(env_action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            ret_to_go = torch.cat([ret_to_go, torch.zeros(1, device=device)])
            ret_to_go[-1] = ret_to_go[-2] - reward
            actions[-1] = env_action
            timesteps = torch.cat([timesteps, torch.tensor([i+1], device=device)])

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length


def main():
    _, run_type = sys.argv
    set_seed()
    env = gym.make('CartPole-v1')
    print(torch.cuda.is_available())

    train_config = {
        "learning_rate": 2e-4,
        "epochs": 15,
        "batch_size": 256,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "warmup_steps": 10000,
        "num_workers": 0
    }

    model_config = {
        "hidden_size": 64,
        "c_len": 20,
        "device": "auto",
        'dropout': 0.1,
    }

    full_config = {**train_config, **model_config}
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    c_len = model_config["c_len"]
    model = DecisionTransformer(state_dim, action_dim, model_config["hidden_size"], c_len, 200, True, n_head=1, n_layer=3,  
            n_inner=4*model_config['hidden_size'],
            n_positions=1024,
            resid_pdrop=model_config['dropout'],
            attn_pdrop=model_config['dropout'], device=model_config["device"])

    if run_type == 'eval' and os.path.exists('models/gpt2_with_pad'):
        model.load_state_dict(torch.load('models/gpt2_with_pad'))
    else:
        wandb.login()
        wandb.init(project='decision-transformer')
        wandb.run.name = 'mingpt'
        wandb.config = full_config


        train_dataset = TrajectoryDataset(c_len, state_dim, action_dim)
        trainer = Trainer(full_config, model, train_dataset, loss_fn=nn.BCELoss())
        trainer.train()
        wandb.run.finish()
    env.close()

    test_env = gym.make('CartPole-v1', render_mode='human')
    with open('dataset/trajectories_metadata.pkl', 'rb') as f:
        ds_mean, ds_std = pickle.load(f)

    print(Trainer.evaluate(model, test_env, ds_mean, ds_std, full_config['device']))
    test_env.close()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set as {seed}")

if __name__ == '__main__':
    main()