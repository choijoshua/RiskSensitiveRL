import torch
import torch.nn as nn
import numpy as np
import time
import os
from scipy.stats import norm

from torch.optim import Adam
from torch.distributions import MultivariateNormal

from replaybuffer import ReplayBuffer
from noise import OrnsteinUhlenbeckActionNoise
from network import ActorNetwork, CriticNetwork

import gym

class DistributionalDDPG:

    def __init__(self, env, actor_class, critic_class, **hyperparameters):
        # initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # extract env info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # initialize actor and critic networks
        self.actor = actor_class(self.obs_dim, self.act_dim)
        self.critic = critic_class(self.obs_dim, self.act_dim)

        self.target_actor = actor_class(self.obs_dim, self.act_dim)
        self.target_critic = critic_class(self.obs_dim, self.act_dim)

        # initialize actor and critic optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.critic_lr, weight_decay=self.weight_decay)

        # initialize noise for exploration
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim))

        # initialize replaybuffer
        self.memory = ReplayBuffer(self.buffer_capacity)
        self.initial_memory()
        
        # initialize weights for target networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        # help print out the summary of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
            'episode': 0,           # episode number
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_loss': [],       # losses of the actor network in current iteration
            'critic_loss': [],       # losses of the critic network in current iteration
		}

    def learn(self, total_timesteps):
        
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"Running for a total of {total_timesteps} timesteps")

        t = 0
        self.returns = []
        self.actor_loss = []
        self.critic_loss = []

        while t < total_timesteps:
            ep_len = 0
            ep_rews = 0
            obs = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.noise.reset()
            done = False
            # sample alpha from a uniform distribution of (0, 1)
            self.alpha = torch.rand(1).unsqueeze(0)

            self.actor.train()
            self.critic.train()

            for ep_t in range(1, self.max_timesteps_per_episode+1):

                t += 1
                self.logger['t_so_far'] = t

                action = self.get_action(obs)
                next_obs, reward, done, _ = self.env.step(torch.squeeze(action, dim=0))

                next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward]).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                ep_rews += reward.item()
                ep_len += 1
                self.memory.insert([obs, action, reward, mask, next_obs, self.alpha])

                if done:
                    break

                if t == total_timesteps:
                    break

                batch_obs, batch_action, batch_rew, batch_done, batch_next_obs, batch_alpha = self.memory.sample(self.batch_size)

                # calculate Q_next distribution
                batch_next_action = self.target_actor(batch_next_obs, batch_alpha)
                mu, sig = self.target_critic(batch_next_obs, batch_next_action, batch_alpha)
                mu_t = self.gamma * mu * (1-batch_done) + batch_rew
                sig_t = self.gamma**2 * sig * (1-batch_done)

                mu_t = mu_t.detach()
                sig_t = sig_t.detach()

                mu_p, sig_p = self.critic(batch_obs, batch_action, batch_alpha)

                # Compute Critic Loss 
                closs = torch.pow(torch.abs(mu_t-mu_p), 2)+ (sig_t+sig_p-2*torch.sqrt(sig_p*sig_t))
                critic_loss = closs.mean()

                torch.autograd.set_detect_anomaly(True)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optim.step()

                # Compute Actor Loss
                mu_a, sig_a = self.critic(batch_obs, self.actor(batch_obs, batch_alpha), batch_alpha)
                n = torch.tensor(norm.pdf(norm.ppf(batch_alpha.detach().numpy())))

                actor_loss = -(mu_a - n * torch.sqrt(sig_a)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.soft_update(self.target_actor, self.actor, self.tau)
                self.soft_update(self.target_critic, self.critic, self.tau)

                self.logger['actor_loss'].append(actor_loss.detach())
                self.logger['critic_loss'].append(critic_loss.detach())

                obs = next_obs

            self.logger['episode'] += 1
            self.logger['batch_lens'].append(ep_len)
            self.logger['batch_rews'].append(ep_rews)
            # record rewards and losses every episode
            self.returns.append(ep_rews)
            self.actor_loss.append(np.mean([losses.float().mean() for losses in self.logger['actor_loss']]))
            self.critic_loss.append(np.mean([losses.float().mean() for losses in self.logger['critic_loss']]))

            if self.logger['episode'] % self.log_interval == 0:
                self._log_summary()

            if self.logger['episode'] % self.save_freq == 0:
                self.save()

    def get_action(self, obs):
        self.actor.eval()
        mean_action = self.actor(obs, self.alpha)
        self.actor.train()
        mean_action = mean_action.data

        mean_action += torch.tensor(self.noise.noise()).unsqueeze(0)

        return self.action_norm(mean_action)
    
    def action_norm(self, action):
        action = (action + 1) / 2  
        action *= (self.env.action_space.high - self.env.action_space.low)
        action += self.env.action_space.low
        return action

    # util function
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # util function
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # util function
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
                self.alpha = torch.rand(1).unsqueeze(0)
            
            # Randomly select an action
            action = self.get_action(obs)
            next_obs, reward, done, _ = self.env.step(torch.squeeze(action, dim=0))
            reward = torch.tensor([reward]).unsqueeze(0)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
            
            # Store new experience into memory
            self.memory.insert([obs, action, reward, done, next_obs, self.alpha])
            obs = next_obs


    def _init_hyperparameters(self, hyperparameters):

        self.batch_size = 4
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.99
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.tau = 0.001
        self.weight_decay = 1e-2
        self.alpha = torch.rand(1).unsqueeze(0)

        self.save_freq = 8                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.log_interval = 8
        self.buffer_capacity = 1000000
        self.initial_buffer_capacity = 200


        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def save(self):
        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)
        actor_model_path = os.path.join(save_dir, f'actor_model.pth')
        torch.save(self.actor.state_dict(), actor_model_path)
        critic_model_path = os.path.join(save_dir, f'critic_model.pth')
        torch.save(self.critic.state_dict(), critic_model_path)
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
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_loss']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_loss']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 4))
        avg_critic_loss = str(round(avg_critic_loss, 4))

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
        print(f"|    actor_lr         | {self.actor_lr}", flush=True)
        print(f"|    actor_loss       | {avg_actor_loss}", flush=True)
        print(f"|    critic_loss      | {avg_critic_loss}", flush=True)
        print(f"--------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_loss'] = []
        self.logger['critic_loss'] = []


env = gym.make("Pendulum-v0")
model = DistributionalDDPG(env=env, actor_class=ActorNetwork, critic_class=CriticNetwork)
model.learn(10)