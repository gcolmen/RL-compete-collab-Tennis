import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, act_size = 1, actor=False, seed=2):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        
        self.seed = torch.manual_seed(seed)
        
        if actor :
            self.fc1 = nn.Linear(input_dim, hidden_in_dim)
            self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        else :
            self.fc1 = nn.Linear(input_dim, hidden_in_dim)
            self.fc2 = nn.Linear(hidden_in_dim + act_size * 2, hidden_out_dim)
        
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        #self.fc4 = nn.Linear(256, output_dim)
        self.nonlin = nn.LeakyReLU(0.01) #leaky_relu
        self.tanh = torch.tanh
        
        self.bn_input = nn.BatchNorm1d(hidden_in_dim)
        self.bn_hid = nn.BatchNorm1d(output_dim)

        #Dropout
        self.dpout = nn.Dropout(p=0.30)
        
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, act=None):
        if self.actor:
            # return a vector of the force
            x = self.fc1(x)
            x = self.bn_input(x)
            x = self.nonlin(x)
#             bn2 = self.bn_hid(self.fc2(h1))
            x = self.nonlin(self.fc2(x))
            x = self.fc3(x)
            x = self.bn_hid(x)
            #x = self.fc4(x)

            return self.tanh(x)
        else:
            # critic network simply outputs a number
            #bn = self.bn_input(self.fc1(x))
            x = self.fc1(x)
            x = self.bn_input(x)
            x = self.nonlin(x)
            x = self.dpout(x)
            x = torch.cat((x, act), dim=1)
#             bn2 = self.bn_hid(self.fc2(h1))
            x = self.nonlin(self.fc2(x))
            x = self.dpout(x)
            x = self.fc3(x)
            #x = self.dpout(x)
            #x = self.fc4(x)

            return x