import os, sys
import gym
from a2c_model_torch import ActorCritic
import torch
import torch.optim as optim
import numpy as np
import pickle
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_returns(final_value, rewards, masks, gamma=0.99):
    total_reward = final_value
    discounted_rewards = [0 for _ in range(len(rewards))]
    for step in reversed(range(len(rewards))):
        total_reward = gamma * total_reward * masks[step] + rewards[step]
        discounted_rewards[step] = total_reward
    return discounted_rewards

def train_model(env, model, optimizer, n_iters):
    trajectories = []
    for episode in range(n_iters):
        log_probs = []
        values = []
        rewards = []
        masks = []
        # traj_number = episode * 30
        # if traj_number != 0 and traj_number % 150 == 0:
        #     with open(f"dataset/traj_{traj_number - 150}-{traj_number}.pkl", "wb") as dataset_file:
        #         pickle.dump(trajectories, dataset_file)
        #     trajectories = []
        # recordTrajectories(env, model, trajectories)
        recordTrajectories(env, model, trajectories)
        state, _info = env.reset()

        for step in range(200):
            state = torch.from_numpy(np.array(state)).to(device)
            dist, value = model(state.unsqueeze(0))
            action = dist.sample()
            next_state, reward, done, _truncated, _info = env.step(action.cpu().numpy()[0])

            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state
            
            if done:
                print(f'Iteration: {episode}, Score: {step}')
                break

        final_value = 0
        if step == 199:
            final_state = torch.FloatTensor(state).to(device)
            _, final_value = model(final_state.unsqueeze(0))
        returns = compute_returns(final_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item(), "return": step})
    torch.save(model, 'a2c_model')
    with open(f"dataset/trajectories.pkl", "wb") as dataset_file:
        pickle.dump(trajectories, dataset_file)
    env.close()

def recordTrajectories(env, model, trajectories, num_episodes=20):
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _info = env.reset()
            trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}
            for _ in range(200):
                state = torch.from_numpy(np.array(state)).unsqueeze(0).to(device)
                trajectory["states"].append(state)
                dist, _ = model(state)
                action = dist.sample()
                state, reward, done, _truncated, _info = env.step(action.cpu().numpy()[0])
                trajectory["actions"].append(np.array([1 if i == action else 0 for i in range(env.action_space.n)]))
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
            trajectories.append(trajectory)



def testModel(model, num_episodes):
    env = gym.make("CartPole-v1", render_mode='human')
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _info = env.reset()
            for _ in range(200):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, _ = model(state)
                action = dist.sample()
                state, _reward, _done, _truncated, _info = env.step(action.cpu().numpy()[0])
    env.close()

def main():
    if len(sys.argv) != 3:
        print('Please input either test or train and the number of episodes to test')
        return
    

    num_hidden = 128
    _, run_type, num_episodes = sys.argv
    print("running on", device)

    if run_type == 'test' and os.path.exists('a2c_model'):
        model = torch.load('a2c_model')
        testModel(model, int(num_episodes))
    elif run_type == 'trajectories' and os.path.exists('a2c_model'):
        model = torch.load('a2c_model')
        trajectories = []
        env = gym.make("CartPole-v1")
        recordTrajectories(env, model, trajectories, int(num_episodes))
        with open('dataset/traj_dataset_small.pkl', 'wb') as f:
            pickle.dump(trajectories, f)
        env.close()
    else:
        wandb.login()
        wandb.init(project='a2c')
        wandb.run.name = 'Adam'

        env = gym.make("CartPole-v1")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        model = ActorCritic(state_size, action_size, num_hidden).to(device)
        optimizer = optim.Adam(model.parameters())
        train_model(env, model, optimizer, 700)

        env.close()
        wandb.run.finish()

if __name__ == '__main__':
    main()
