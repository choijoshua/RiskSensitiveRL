import torch
import torch.nn as nn
import numpy as np
import time
import os

from torch.optim import Adam

from replaybuffer import ReplayBuffer


class RiskSensitiveDQN:
    def __init__(self, model_class, env, **hyperparameters):
        # initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # extract env info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        
        # initialize target network and q-network
        self.target_network = model_class(self.obs_dim, self.act_dim)
        self.q_network = model_class(self.obs_dim, self.act_dim)

        # initialize actor and critic optimizers
        self.target_optim = Adam(self.target_network.parameters(), lr = self.lr)
        self.q_network_optim = Adam(self.q_network.parameters(), lr = self.lr)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # initialize replaybuffer
        self.memory = ReplayBuffer(self.buffer_capacity)
        self.initial_memory()

        # help print out the summary of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
            'episode': 0,           # episode number
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'losses': [],           # losses of the q network in current iteration
		}

    def learn(self, total_timesteps):

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"Running for a total of {total_timesteps} timesteps")

        t = 0
        self.returns = []
        self.losses = []

        while t < total_timesteps:
            ep_len = 0
            ep_rews = 0
            obs = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            done = False

            if self.logger['episode'] > 400:
                self.kappa = -0.4
            elif self.logger['episode'] > 200:
                self.kappa = -0.2

            for ep_t in range(1, self.max_timesteps_per_episode+1):
                t += 1
                ep_len += 1
                self.logger['t_so_far'] = t

                action = self.get_action(obs)
                next_obs, reward, done, _ = self.env.step(action.item())

                next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                done = torch.tensor([done]).unsqueeze(0)

                ep_rews += reward.item()
                self.memory.insert([obs, action, reward, done, next_obs])

                if done:
                    break

                batch_obs, batch_action, batch_rew, batch_done, batch_next_obs = self.memory.sample(self.batch_size)

                self.q_network.train()

                Q_target = batch_rew + ~batch_done * self.gamma * torch.max(self.target_network(batch_next_obs), dim=-1, keepdim=True)[0]
                Q_curr = self.q_network(batch_obs).gather(1, batch_action)

                kappa = []
                td_error = Q_target - Q_curr
                for i in range(len(td_error)):
                    if td_error[i] > 0:
                        kappa.append(1 - self.kappa)
                    else:
                        kappa.append(1 + self.kappa)

                kappa = torch.tensor(kappa, dtype=torch.float32).unsqueeze(1)

                loss = nn.MSELoss()(kappa * Q_curr, kappa * Q_target)

                self.q_network_optim.zero_grad()
                loss.backward()
                self.q_network_optim.step()

                if t % self.target_update_interval:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                obs = next_obs

                self.logger['losses'].append(loss.detach())

            self.returns.append(ep_rews)
            self.losses.append(np.mean([losses.float().mean() for losses in self.logger['losses']]))

            self.logger['episode'] += 1
            self.logger['batch_lens'].append(ep_len)
            self.logger['batch_rews'].append(ep_rews)

            if self.logger['episode'] % self.log_interval == 0:
                self._log_summary()

            if self.logger['episode'] % self.save_freq == 0:
                self.save()

    def get_action(self, obs):

        self.q_network.eval()
        if np.random.random_sample() > self.epsilon:
            return torch.tensor([self.q_network(obs).argmax()], dtype=torch.long).unsqueeze(0)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def initial_memory(self):
        # Initialize replay memory with a burn-in number of episodes/transitions.
        done = False
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Iterate until we store "burn_in" buffer
        for i in range(self.initial_buffer_capacity):
            # Reset environment if done
            if done:
                obs = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Randomly select an action
            action = torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
            next_obs, reward, done, _ = self.env.step(action.item())
            reward = torch.tensor([reward]).unsqueeze(0)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            done = torch.tensor([done]).unsqueeze(0)
            
            # Store new experience into memory
            self.memory.insert([obs, action, reward, done, next_obs])
            obs = next_obs

    def _init_hyperparameters(self, hyperparameters):

        self.batch_size = 64
        self.max_timesteps_per_episode = 500
        self.gamma = 0.99
        self.epsilon = 0.05
        self.kappa = 0.0
        self.lr = 0.001

        self.render = False
        self.save_freq = 10
        self.log_interval = 4
        self.target_update_interval = 1000
        self.buffer_capacity = 100000
        self.initial_buffer_capacity = 500

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    
    def save(self):
        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'model.pth')
        torch.save(self.q_network.state_dict(), model_path)
        print(f"Model saved")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values.
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_loss = np.mean([losses.float().mean() for losses in self.logger['losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_loss = str(round(avg_loss, 5))

        episodes = self.logger['episode']

        # Print logging statements
        print(flush=True)
        print(f"--------------------------------", flush=True)
        print(f"| rollout/            |")
        print(f"|    ep_len_mean      | {avg_ep_lens}", flush=True)
        print(f"|    ep_rew_mean      | {avg_ep_rews}", flush=True)
        print(f"| time/               |")
        print(f"|    episodes         | {episodes}", flush=True)
        print(f"|    time_elapsed     | {delta_t}", flush=True)
        print(f"|    total_timeteps   | {t_so_far}", flush=True)
        print(f"| train/              |")
        print(f"|    learning_rate    | {self.lr}", flush=True)
        print(f"|    loss             | {avg_loss}", flush=True)
        print(f"|    kappa            | {self.kappa}", flush=True)
        print(f"--------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['losses'] = []


    
