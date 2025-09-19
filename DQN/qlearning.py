import torch
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm
from cliffwalking import CliffWalkingEnv

import gym
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, env, kappa=0):

        # extract env info
        self.env = env
        self.obs_dim = env.observation_space.n
        self.act_dim = env.action_space.n
        
        # initialize target network and q-network
        self.q_table = np.zeros((self.obs_dim, self.act_dim))

        self.gamma = 0.99
        self.epsilon = 0.05
        self.kappa = kappa
        self.alpha = 0.2

    def learn(self, total_episodes):

        returns = []
        ep_lens = []

        # Iterate over 500 episodes
        for _ in tqdm(range(total_episodes)):
            state = self.env.reset()
            done = False
            ep_rews = 0
            ep_len = 0

            # While episode is not over
            while not done:
                # Choose action        
                action = self.egreedy_policy(self.q_table, state)
                # Do the action
                next_state, reward, done, _ = self.env.step(action)
                # Update q_values        
                td_target = reward + self.gamma * np.max(self.q_table[next_state])
                td_error = td_target - self.q_table[state][action]
                kappa = 1
                if td_error > 0:
                    kappa = 1 - self.kappa
                else:
                    kappa = 1 + self.kappa

                self.q_table[state][action] += kappa * self.alpha * td_error
                # Update state
                state = next_state

                ep_rews += reward
                ep_len += 1

            returns.append(ep_rews)
            ep_lens.append(ep_lens)

        return returns, ep_lens


    def egreedy_policy(self, q_values, state):  
        # Get a random number from a uniform distribution between 0 and 1,
        # if the number is lower than epsilon choose a random action
        if np.random.random() < self.epsilon:
            return np.random.choice(4)
        # Else choose the action with the highest value
        else:
            return np.argmax(q_values[state])
        

def get_optimal_action(q_table, row, col):
    state = row * 12 + col
    return np.argmax(q_table[state])

def visualize_policy(q_table):
    rows = 4
    cols = 12
    actions = ['^', '>', 'v', '<'] 

    for row in range(rows):
        for col in range(cols):
            optimal_action = get_optimal_action(q_table, row, col)
            print(actions[optimal_action], end=' ')
        print()



env = CliffWalkingEnv()

model = QLearning(env=env, kappa=0.9)
returns, ep_lens = model.learn(total_episodes=500)

visualize_policy(model.q_table)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(returns)), returns, label='Episode Return')
plt.xlabel('Episode')
plt.ylabel('Episode Return')
plt.title('Episode Return over Time')
plt.legend()
plt.grid(True)
plt.show()
