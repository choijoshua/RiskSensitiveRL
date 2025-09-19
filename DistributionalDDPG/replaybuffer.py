import torch
import random

class ReplayBuffer:
  def __init__(self, capacity=1000000):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def insert(self, transition):
    if len(self.memory) < self.capacity:
      self.memory.append(None)

    self.memory[self.position] = transition
    self.position = (self.position+1)%self.capacity

  def sample(self, batch_size):
    assert self.can_sample(batch_size)

    batch = random.sample(self.memory, batch_size)
    batch = zip(*batch)
    return [torch.cat(items) for items in batch]

  def can_sample(self, batch_size):
    return batch_size <= len(self.memory)

  def __len__(self):
    return len(self.memory)