import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .parameters import Parameters
EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    def __init__(self, args: Parameters):
        super(Critic, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.fcs1 = nn.Linear(self.state_dim,256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1 = nn.Linear(self.action_dim,128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128,1)
        self.fc3.weight.data.uniform_(-EPS,EPS)
    

    def forward(self, state, action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2,a1),dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, args: Parameters):
        super(Actor, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.fc1 = nn.Linear(args.state_dim,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4 = nn.Linear(64, args.action_dim)
        self.fc4.weight.data.uniform_(-EPS,EPS)

    def forward(self, state):
        print("=====")
        print(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.sigmoid(self.fc4(x))
        return action



