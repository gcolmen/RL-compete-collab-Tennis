# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utils import hard_update ##, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent:
    def __init__(self, in_actor, out_actor, in_critic, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        hidden_in_actor = 64
        hidden_out_actor = 128
        hidden_in_critic = hidden_in_actor
        hidden_out_critic = hidden_out_actor

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, out_actor, actor=False).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, out_actor, actor=False).to(device)

        self.noise = OUNoise(out_actor, scale=0.9) #scale 1.0
        self.noise_shape = out_actor
        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        WD = 1e-5
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=WD)

    def reset(self):
        self.noise.reset()
    
    def noisef(self, mean=0, sigma=0.08) :
        return np.random.normal(mean, sigma, self.noise_shape)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs).cpu().data.numpy() + noise * self.noisef() #self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs).cpu() # + noise * self.noisef() #self.noise.noise()
        return action
