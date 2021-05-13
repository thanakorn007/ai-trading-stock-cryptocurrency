import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .nn_model import NeuralNetwork
from .memory import ExperienceReplayMemory



class DQNAgent:
    '''Agent'''

    def __init__(self,
                 input_sz,
                 action_sz,
                 discount_factor=0.9,
                 epsilon=1,
                 epsilon_min=0.001,
                 epsilon_decay=0.995,
                 batch_size=128,
                 lr=0.001,
                 ticker = '',
                 capacity=5000,
                 layers=[40, 50]):

        # network for predict q values (input = state vector, output = q values dim action size)
        self.model = NeuralNetwork(input_sz, action_sz, layers=layers)
        # memory for experience replay
        self.memory = ExperienceReplayMemory(capacity=capacity)
        self.discount_factor = discount_factor  # discount factor (gamma)
        self.reward_window = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr) #Adam optimizer learning rate default = 0.001
        self.last_state = torch.Tensor(input_sz).unsqueeze(0)  # shape [1, 5]
        self.last_action = 0
        self.last_reward = 0

        # epsilon greedy selection
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_epsilon = []

        self.batch_size = batch_size #batch size
        self.action_sz = action_sz
        self.criterion = nn.MSELoss() #Mean squared error loss
        self.losses = []
        self.stock_name = ticker



    def get_action(self, state):
        '''
        Epsilon-greedy selection
        '''

        # if np.random.rand() <= self.epsilon:
        #     # if random samples from a uniform distribution over [0, 1) less than epsilon
        #     return np.random.choice(self.action_sz) # return random choice of action space
        # else return action maximum q values from network
        q_values = self.model(state.clone().detach())
        q_values_prob = F.softmax(q_values * 1000, dim=0)
        action = q_values_prob.multinomial(num_samples=1)
        return action.data[0, 0]

    def train(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state) #output network
        #outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_outputs = self.model(batch_next_state) # Q(s',a')
        argmax_target = torch.argmax(next_outputs, dim=1) # argmax q values from Q(s',a')
        target = batch_reward + self.discount_factor * next_outputs.max(1)[0]  # target = reward + gamma*Q(s',a')
        target_full = torch.zeros_like(outputs)
        target_full.copy_(outputs)
        # replace maximum q values from output with target
        for i in range(len(batch_state)):
            target_full[i][argmax_target[i]] = target[i]

        # mse loss between output and target full
        loss = self.criterion(outputs, target_full)

        assert np.isnan(loss.item()) == False
        if np.isnan(loss.item()):
            print('=========stop=========')
            print('\n')

        self.losses.append(loss.item())  # loss

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, reward, new_state):
        new_state = torch.tensor(new_state[0]).float().unsqueeze(0) #state vector
        # push (last state, new state, last action, last reward into memory)
        self.memory.push((self.last_state,
                          new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        # get action
        action = self.get_action(new_state)

        if len(self.memory.memory) > self.batch_size:
            # sample batch event from memory (experience replay)
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
            # learn
            self.train(batch_state, batch_next_state, batch_action, batch_reward)

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.save_epsilon.append(self.epsilon)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # add reward to window reward
        if len(self.reward_window) > self.batch_size:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'state_dict/' + self.stock_name + '.pth')
        print('=> Saved!!!...')

    def load(self):
        if os.path.isfile('state_dict/' + self.stock_name + '.pth'):
            print("loading state dict form " + self.stock_name)
            checkpoint = torch.load('state_dict/' + self.stock_name + '.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.eval()
            print('Completed!')
        else:
            print("no parameters found...")
