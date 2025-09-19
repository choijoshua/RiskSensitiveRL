import gym
import sys
import torch
import numpy as np

from dqn import RiskSensitiveDQN
from network import FeedForwardNN

import argparse
import matplotlib.pyplot as plt

def train(env, hyperparameters, qlearning_model, timesteps, graph_=1):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			network - the model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for DQN
	model = RiskSensitiveDQN(model_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing model to continue training on
	if qlearning_model != '':
		print(f"Loading in {qlearning_model}...", flush=True)
		model.q_network.load_state_dict(torch.load(qlearning_model))
		print(f"Successfully loaded.", flush=True)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the q-network model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=timesteps)
	if graph_==1:
		graph(model)
	
def test(env, qlearning_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			qlearning_model - the neural network model to load in

		Return:
			None
	"""
	print(f"Testing {qlearning_model}", flush=True)

	# If the actor model is not specified, then exit
	if qlearning_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	qnetwork = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	qnetwork.load_state_dict(torch.load(qlearning_model))

def graph(agent):

	returns = agent.returns
	loss = agent.losses

	# Plot episode returns (total rewards)
	plt.figure(figsize=(12, 6))
	plt.plot(np.arange(len(returns)), returns, label='Episode Return')
	plt.xlabel('Episode')
	plt.ylabel('Episode Return')
	plt.title('Episode Return over Time')
	plt.legend()
	plt.grid(True)
	plt.show()

	# Plot actor and critic losses
	plt.figure(figsize=(12, 6))
	plt.plot(np.arange(len(loss)), agent.losses, label='Actor Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Losses over Iterations')
	plt.legend()
	plt.grid(True)
	plt.show()


def main(args):

    hyperparameters = {
				'batch_size': 32, 
				'max_timesteps_per_episode': 400, 
				'gamma': 0.95, 
				'epsilon': 0.05,
				'kappa': -0.0,
				'lr': 0.001,
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
    environments = ["CartPole-v1", "LunarLander-v2", "Acrobot-v1", "MountainCar-v0"]
    env = gym.make(environments[3])

    # Train or test, depending on the mode specified
    if args.train == 1:
        train(env=env, hyperparameters=hyperparameters, qlearning_model=args.model, timesteps=args.t, graph_=args.graph)
    else:
        test(env=env, qlearning_model=args.model)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', dest='train', type=int, default=1)              # can be 'train' or 'test'
	parser.add_argument('--weight', dest='model', type=str, default='')      	  # your model filename
	parser.add_argument('--timesteps', dest='t', type=int, default=1000)
	parser.add_argument('--graph', dest='graph', type=int, default=1)   

	args = parser.parse_args()

	main(args)

