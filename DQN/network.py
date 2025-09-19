import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        
        self.layer1 = nn.Linear(in_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.layer2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.layer3 = nn.Linear(300, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        activation1 = F.relu(self.ln1(self.layer1(obs)))
        activation2 = F.relu(self.ln2(self.layer2(activation1)))
        output = self.layer3(activation2)

        return output