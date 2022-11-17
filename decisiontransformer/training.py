import torch
import numpy as np
import time
import gym
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from decision_transformer import DecisionTransformer
from dataset.trajectory_dataset import TrajectoryDataset

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

        self.iter_num = 0
        self.iter_time = 0.0
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/config['warmup_steps'], 1)
        )
        # self.optimizer = self.model.configure_optimizers(config)
        self.loss_fn = loss_fn

    def train(self):
        model, config = self.model, self.config
        train_losses = []
        logs = dict()
        train_start = time.time()

        model.train()
        print('train start')
        for epoch in tqdm(range(config['epochs'])):
            train_loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

            batch_losses = []
            for batch_num, sample in enumerate(tqdm(train_loader)):
                states, actions, returns, dones, timesteps, attn_mask = sample
                states = states.to(dtype=torch.float, device=self.device)
                actions = actions.to(dtype=torch.float, device=self.device)
                returns = returns.to(dtype=torch.float, device=self.device)
                timesteps = timesteps.to(dtype=torch.long, device=self.device)

                action_target = torch.clone(actions)


                action_preds = self.model.forward(
                    states, actions, returns, timesteps=timesteps, attn_mask=attn_mask
                )

                loss = self.loss_fn(action_preds, action_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                batch_losses.append(loss.detach().cpu().item())
                wandb.log({'loss': loss.item()})
            train_losses.extend(batch_losses)
            print(np.mean(np.asarray(batch_losses)))
        
        self.train_losses = train_losses
        logs['time/training'] = time.time() - train_start

    def evaluate(self, env, state_mean, state_std, target_return=200):
        model, device = self.model, self.device
        model.eval()
        model.to(device=device)
        state_dim, act_dim = env.observation_space.shape[0], env.action_space.n

        # state_mean = torch.from_numpy(state_mean).to(device=device)
        # state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        ret_to_go = torch.zeros(1, device=device, dtype=torch.float32)
        ret_to_go[0] = target_return
        timesteps = torch.from_numpy(np.arange(200)).to(device=device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for t in timesteps:

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            norm_state = states.to(dtype=torch.float32) # (states.to(dtype=torch.float32) - state_mean) / state_std

            action = model.get_action(
                norm_state,
                actions.to(dtype=torch.float32),
                ret_to_go.to(dtype=torch.float32),
                timesteps
            )
            action = action.detach().cpu().numpy()
            env_action = np.argmax(action)
            state, reward, done, _ = env.step(env_action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            ret_to_go = torch.cat([ret_to_go, torch.zeros(1, device=device)])
            ret_to_go[-1] = ret_to_go[-2] - reward

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length


def main():
    # wandb.login()
    # env = gym.make('CartPole-v0')
    # print(torch.cuda.is_available())

    # with wandb.init(project='decision-transformer'):
    #     wandb.config = {
    #         "learning_rate": 2e-4,
    #         "epochs": 1000,
    #         "batch_size": 64,
    #         "hidden_size": 128,
    #         "c_len": 4,
    #         "device": "auto",
    #         "weight_decay": 0.01,
    #         "betas": (0.9, 0.999),
    #         "num_workers": 2
    #     }
    #     config = wandb.config
    #     env = gym.make('CartPole-v0')

    #     state_dim = env.observation_space.shape[0]
    #     action_dim = env.action_space.n
    #     c_len = config["c_len"]

    #     train_dataset = TrajectoryDataset('dataset', c_len, state_dim, action_dim)
    #     # train_dataset = Subset(train_dataset, [0])
    #     model = DecisionTransformer(state_dim, action_dim, config["hidden_size"], c_len, 200, True, n_head=1, n_layer=3, device=config["device"])
    #     trainer = Trainer(config, model, train_dataset, loss_fn=lambda a_hat, a: torch.mean((a_hat - a)**2))
    #     trainer.train()
    wandb.login()
    env = gym.make('CartPole-v0')
    print(torch.cuda.is_available())

    wandb.init(project='decision-transformer')
    wandb.run.name = 'gpt2-colab-full-clen-20-lr-1e-4'
    wandb.config = {
        "learning_rate": 1e-3,
        "epochs": 500,
        "batch_size": 64,
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
    config = wandb.config
    env = gym.make('CartPole-v0')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    c_len = config["c_len"]

    train_dataset = TrajectoryDataset('dataset', c_len, state_dim, action_dim)
    model = DecisionTransformer(state_dim, action_dim, config["hidden_size"], c_len, 200, True, n_head=1, n_layer=3, n_inner=4*config['hidden_size'],
            activation_function=config['activation_function'],
            n_positions=1024,
            resid_pdrop=config['dropout'],
            attn_pdrop=config['dropout'], device=config["device"])
    trainer = Trainer(config, model, train_dataset, loss_fn=lambda a_hat, a: torch.mean((a_hat - a)**2))
    trainer.train()
    wandb.run.finish()

if __name__ == '__main__':
    main()