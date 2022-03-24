import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorCritic, self).__init__()
        self.obs_size = obs_size
        self.act_size = act_size

        self.feature_layer = nn.Sequential(nn.Linear(self.obs_size, 128), nn.ReLU(),
                                           nn.Linear(128, 128), nn.ReLU())

        self.policy_layer = nn.Linear(128, self.act_size)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, obs):
        feature = self.feature_layer(obs)
        policy = self.policy_layer(feature)
        policy = F.softmax(policy, -1)
        value = self.value_layer(feature)
        return policy, value

    def act(self, observation):
        policy, value = self.forward(observation)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action
