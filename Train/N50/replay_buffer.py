import numpy as np
import torch

class ReplayBuffer:

    def __init__(self, size, obs_size):
        '''self.size = size
        self.obs = np.zeros([self.size, obs_size], dtype=np.float32)
        self.next_obs = np.zeros([self.size, obs_size], dtype=np.float32)
        self.actions = np.zeros([self.size], dtype=np.int32)
        self.rewards = np.zeros([self.size], dtype=np.float32)
        self.done = np.zeros([self.size], dtype=np.bool)
        self.num_in_buffer = 0#当前经验库里已有经验数量
        self.next_idx = 0#下一个经验来的时候应该存放的位置'''
        self.size = size
        self.obs = torch.zeros([self.size, obs_size]).to('cuda:3')
        self.next_obs = torch.zeros([self.size, obs_size]).to('cuda:3')
        self.actions = torch.zeros([self.size]).to('cuda:3')
        self.rewards = torch.zeros([self.size]).to('cuda:3')
        self.done = torch.zeros([self.size,1],dtype=int).to('cuda:3')
        self.num_in_buffer = 0  # 当前经验库里已有经验数量
        self.next_idx = 0  # 下一个经验来的时候应该存放的位置


    def store_transition(self, obs, action, reward, next_obs, done):
        self.obs[self.next_idx] = obs
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward

        self.next_obs[self.next_idx] = next_obs
        self.done[self.next_idx] = done

        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        self.next_idx = (self.next_idx + 1) % self.size

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions
         can be sampled from the buffer."""
        return batch_size + 1<= self.num_in_buffer

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = np.random.choice(self.num_in_buffer, batch_size,replace=False)
        return self.obs[idxes], \
               self.actions[idxes], \
               self.rewards[idxes], \
               self.next_obs[idxes], \
               self.done[idxes]

