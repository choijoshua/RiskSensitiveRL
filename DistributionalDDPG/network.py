import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorNetwork, self).__init__()

        # Layer 1
        self.linear1 = nn.Linear(obs_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        # Layer 2
        self.linear2 = nn.Linear(400 + 1, 300)
        self.bn2 = nn.BatchNorm1d(300)

        # Output Layer
        self.mu = nn.Linear(300, act_dim)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, obs, alpha):
        x = obs

        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(torch.cat((x, alpha), 1))
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        return torch.tanh(self.mu(x))


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticNetwork, self).__init__()

        # Layer 1
        self.linear1 = nn.Linear(obs_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.linear2 = nn.Linear(400 + act_dim, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.linear3 = nn.Linear(300 + 1, 300)

        # Output layer (single value)
        self.mean = nn.Linear(300, 1)
        self.var = nn.Linear(300, 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.linear3.weight)
        fan_in_uniform_init(self.linear3.bias)

        nn.init.uniform_(self.mean.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mean.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        nn.init.uniform_(self.var.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.var.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, obs, action, alpha):
        x = obs

        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, action), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.relu(self.linear3(torch.cat((x, alpha), 1)))

        # Predict mean and variance of future returns
        mean = self.mean(x)
        var = F.softplus(self.var(x))  # Softplus activation for positive variance

        return mean, var