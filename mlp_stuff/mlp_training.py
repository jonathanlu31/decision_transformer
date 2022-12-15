import torch
import torch.nn as nn
import random
import numpy as np
import gym
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from mlpmodel import FeedForward
from dataset.trajectory_dataset import TrajectoryDataset

class Trainer:
    def __init__(self, config, model, datasets, loss_fn):
        self.config = config
        self.model = model
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
        self.scaler = torch.cuda.amp.GradScaler()
        self.train_loader = DataLoader(
                datasets[0],
                shuffle=True,
                pin_memory=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )
        self.valid_loader = DataLoader(
                datasets[1],
                shuffle=True,
                pin_memory=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

    def train(self):
        model, config = self.model, self.config
        train_losses = []
        train_step = 0
        valid_step = 0

        print('train start')
        for epoch in tqdm(range(config['epochs'])):
            batch_losses = []
            model.train()
            for batch_num, sample in enumerate(tqdm(self.train_loader)):
                states, actions, returns = sample
                states = states.to(dtype=torch.float, device=self.device)
                actions = actions.to(dtype=torch.float, device=self.device)
                self.optimizer.zero_grad()
                # returns = returns.to(dtype=torch.float, device=self.device)

                action_preds = self.model(states).squeeze()

                loss = self.loss_fn(action_preds, actions)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                wandb.log({'train_step': train_step, 'train_loss': loss.item()})
                train_step += 1
                batch_losses.append(loss.item())
            print(np.mean(np.asarray(batch_losses)))

            if epoch < 3:
                continue

            with torch.no_grad():
                model.eval()
                for batch_num, sample in enumerate(tqdm(self.valid_loader)):
                    states, actions, returns = sample
                    states = states.to(dtype=torch.float, device=self.device)
                    actions = actions.to(dtype=torch.float, device=self.device)
                    # returns = returns.to(dtype=torch.float, device=self.device)

                    action_preds = self.model(states).squeeze()

                    loss = self.loss_fn(action_preds, actions)
                    wandb.log({'valid_step': valid_step, 'valid_loss': loss.item()})
                    valid_step += 1
            train_losses.extend(batch_losses)
        
        self.train_losses = train_losses


def main():
    set_seed()
    wandb.login()
    env = gym.make('CartPole-v1')
    print(torch.cuda.is_available())

    config = {
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32,
        "hidden_size": 256,
        "device": "auto",
        "weight_decay": 1e-4,
        "num_workers": 0
    }
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = FeedForward(state_dim, action_dim, config['hidden_size'])
    wandb.init(project='decision-transformer')
    wandb.run.name = 'ff'
    wandb.config = config

    ds = TrajectoryDataset(None, state_dim, action_dim)
    train_ds, valid_ds = torch.utils.data.random_split(ds, [0.7, 0.3])
    trainer = Trainer(config, model, (train_ds, valid_ds), loss_fn=nn.BCELoss())
    trainer.train()
    wandb.run.finish()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    main()