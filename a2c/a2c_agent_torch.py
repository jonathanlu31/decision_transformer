import os, sys
import gym
from a2c_model_torch import ActorCritic
import torch
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0")

def compute_returns(final_value, rewards, masks, gamma=0.99):
    total_reward = final_value
    discounted_rewards = [0 for _ in range(len(rewards))]
    for step in reversed(range(len(rewards))):
        total_reward = gamma * total_reward * masks[step] + rewards[step]
        discounted_rewards[step] = total_reward
    return discounted_rewards

def train_model(env, model, optimizer, n_iters):
    trajectories = []
    plt_returns = []
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
        state = env.reset()

        for step in range(200):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state.unsqueeze(0))
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state
            
            if done:
                print(f'Iteration: {episode}, Score: {step}')
                plt_returns.append(step)
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
    torch.save(model, 'a2c_model')
    env.close()
    plt.plot(plt_returns)
    plt.savefig('returns.png')

def recordTrajectories(env, model, trajectories):
    with torch.no_grad():
        for _ in range(30):
            state = env.reset()
            trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}
            for _ in range(200):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                trajectory["states"].append(state)
                dist, _ = model(state)
                action = dist.sample()
                state, reward, done, _ = env.step(action.cpu().numpy()[0])
                trajectory["actions"].append(np.array([1 if i == action else 0 for i in range(env.action_space.n)]))
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
            trajectories.append(trajectory)



def testModel(env, model, num_episodes):
    with torch.no_grad():
        trajectory = []
        recordTrajectories(env, model, trajec)
        for _ in range(num_episodes):
            state = env.reset()
            for _ in range(200):
                env.render()
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, _ = model(state)
                action = dist.sample()
                state, _reward, _done, _ = env.step(action.cpu().numpy()[0])
    env.close()

def main():
    if len(sys.argv) != 3:
        print('Please input either test or train and the number of episodes to test')
        return

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    num_hidden = 128
    _, run_type, num_episodes = sys.argv

    if run_type == 'test' and os.path.exists('a2c_model'):
        model = torch.load('a2c_model')
    else:
        model = ActorCritic(state_size, action_size, num_hidden).to(device)
        optimizer = optim.NAdam(model.parameters())
        train_model(env, model, optimizer, 600)

    testModel(env, model, int(num_episodes))

if __name__ == '__main__':
    main()
