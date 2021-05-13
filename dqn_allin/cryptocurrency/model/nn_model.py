import torch
import torch.nn as nn
import torch.nn.functional as F



class NeuralNetwork(nn.Module):
    '''
    Network for predict action trading
    '''

    def __init__(self, input_sz, action_sz, layers=[40, 50]):
        super().__init__()
        self.input_sz = input_sz
        self.action_sz = action_sz
        self.fc1 = nn.Linear(input_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], action_sz)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values