import gym
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from ddpg import DDPG
from network import ActorNetwork, CriticNetwork
import argparse

def train(env, hyperparameters, actor_model, critic_model, timesteps, graph_):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = DDPG(env=env, actor_class=ActorNetwork, critic_class=CriticNetwork, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=timesteps)

	if graph_ == 1:
		graph(model)
	
def rollout(env, actor_model, render=False):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	agent = DDPG(env=env, actor_class=ActorNetwork, critic_class=CriticNetwork)

	# Load in the actor model saved by the PPO algorithm
	agent.actor.load_state_dict(torch.load(actor_model))

	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = agent.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
			obs, rew, done, _ = env.step(torch.squeeze(action, dim=0))

			# Sum all episodic rewards as we go along
			ep_ret += rew.item()
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def test(env, actor_model, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(env, actor_model, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def graph(agent):

	# Plot episode returns (total rewards)
	plt.figure(figsize=(12, 6))
	plt.plot(np.arange(len(agent.returns)), agent.returns, label='Episode Return')
	plt.xlabel('Episode')
	plt.ylabel('Episode Return')
	plt.title('Episode Return over Time')
	plt.legend()
	plt.grid(True)
	plt.show()

	# Plot actor and critic losses
	plt.figure(figsize=(12, 6))
	plt.plot(np.arange(len(agent.actor_loss)), agent.actor_loss, label='Actor Loss')
	plt.plot(np.arange(len(agent.critic_loss)), agent.critic_loss, label='Critic Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Actor and Critic Losses over Iterations')
	plt.legend()
	plt.grid(True)
	plt.show()

def main(args):

    hyperparameters = {
				'batch_size': 64, 
				'max_timesteps_per_episode': 2000, 
				'gamma': 0.99,
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
    environments = ["Pendulum-v0", "LunarLander-v2", "MountainCarContinuous-v0", "BipedalWalker-v2", "InvertedPendulum-v2"]
    env = gym.make(environments[0])

    # Train or test, depending on the mode specified
    if args.train == 1:
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, timesteps=args.t, graph_=args.graph)
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', dest='train', type=int, default=1)              # can be 'train' or 'test'
	parser.add_argument('--timesteps', dest='t', type=int, default=1)
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
	parser.add_argument('--graph', dest='graph', type=int, default=1)   # your critic model filename

	args = parser.parse_args()

	main(args)