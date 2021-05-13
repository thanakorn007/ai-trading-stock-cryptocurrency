import numpy as np
import pandas as pd
import pandas_datareader as web
import talib
import seaborn as sns
import gym
from gym import spaces
from gym.utils import seeding
import enum
import pyfolio as pf
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


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
                 ticker='',
                 capacity=10000,
                 layers=[100, 100, 100]):

        # network for predict q values (input = state vector, output = q values dim action size)
        self.network = self.build_model_nn(input_sz, action_sz, layers=layers)
        self.target_network = self.build_model_nn(input_sz, action_sz, layers=layers)
        self.update_target_network()

        self.discount_factor = discount_factor  # discount factor (gamma)
        self.reward_window = []
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)  # Adam optimizer learning rate default = 0.001
        self.last_state = torch.Tensor(input_sz).unsqueeze(0)  # shape [1, 5]
        self.last_action = 0
        self.last_reward = 0

        # epsilon greedy selection
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_epsilon = []

        self.batch_size = batch_size  # batch size
        self.action_sz = action_sz
        self.criterion = nn.MSELoss()  # Mean squared error loss
        self.losses = []
        self.stock_name = ticker

        # experience replay
        self.capacity = capacity
        self.memory = []

    def seed(self, seeding=101):
        return torch.manual_seed(seeding)

    def build_model_nn(self, input_sz, action_sz, layers):

        class NeuralNetwork(nn.Module):
            '''
            Network for predict action trading
            '''

            def __init__(self, input_sz, action_sz, layers=[100, 100, 100]):
                super().__init__()
                self.input_sz = input_sz
                self.action_sz = action_sz
                self.fc1 = nn.Linear(input_sz, layers[0])
                self.fc2 = nn.Linear(layers[0], layers[1])
                self.fc3 = nn.Linear(layers[1], layers[2])
                self.out = nn.Linear(layers[2], action_sz)

            def forward(self, state):
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                q_values = self.out(x)
                return q_values

        self.seed()

        model = NeuralNetwork(input_sz, action_sz, layers=layers)

        return model

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def get_action(self, state):
        '''
        Epsilon-greedy selection
        '''

        if np.random.rand() <= self.epsilon:
            # if random samples from a uniform distribution over [0, 1) less than epsilon
            return np.random.choice(self.action_sz)  # return random choice of action space

        # else return action maximum q values from network
        q_values = self.network(state.clone().detach())
        q_values_prob = F.softmax(q_values * 10, dim=0)
        action = q_values_prob.multinomial(num_samples=1)
        return action.data[0, 0]

    def train(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.network(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        next_outputs = self.target_network(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.discount_factor * next_outputs  # reward + gamma*Q(s',a')

        loss = F.smooth_l1_loss(outputs, target)  # (1/n) * sum(zi)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, reward, new_state):
        new_state = torch.tensor(new_state[0]).float().unsqueeze(0)  # state vector

        # push (last state, new state, last action, last reward into memory)
        self.push((self.last_state,
                   new_state,
                   torch.LongTensor([int(self.last_action)]),
                   torch.Tensor([self.last_reward])))
        # get action
        action = self.get_action(new_state)

        if len(self.memory) > self.batch_size:
            # sample batch event from memory (experience replay)
            batch_state, batch_next_state, batch_action, batch_reward = self.sample(self.batch_size)
            # learn
            self.train(batch_state, batch_next_state, batch_action, batch_reward)

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.save_epsilon.append(self.epsilon)

        self.update_target_network()

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # add reward to window reward
        if len(self.reward_window) > self.batch_size:
            del self.reward_window[0]
        return action

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        '''
        Sample event from memory (len = batch size)
        '''

        # [batch_state, batch_next_state, batch_action, batch_reward]
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'state_dict/' + self.stock_name + '.pth')
        print('=> Saved!!!...')

    def load(self):
        if os.path.isfile('state_dict/' + self.stock_name + '.pth'):
            print("loading state dict form " + self.stock_name)
            checkpoint = torch.load('state_dict/' + self.stock_name + '.pth')
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.network.eval()
            print('Completed!')
        else:
            print("no parameters found...")


# environment

class Actions(enum.Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class Positions(enum.Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class SingleStockEnv(gym.Env):
    '''
    Single Stock for trading

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, data, capital, trade_fee_bid_percent=0.01, trade_fee_ask_percent=0.005):

        self.trade_fee_bid_percent = trade_fee_bid_percent / 100  # percent
        self.trade_fee_ask_percent = trade_fee_ask_percent / 100  # percent
        self.stock_price_history = data
        self.n_step = self.stock_price_history.shape[0]
        self.capital = capital
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.state_dim = 9  # [stock price, stock_owned, cash_in_hand, rsi, mom, adx, macd, macd_sig, cci]
        self.save_position = []
        self.save_port_val = []
        self.save_cash = []
        self.total_reward = None
        self.stock_value = []
        self.sell_buy = None
        self.save_stock_price = []

        # space
        self.action_space = spaces.Discrete(len(Actions))  # 0, 1, 2

        # episode
        self.start_tick = 0
        self.end_tick = self.n_step - 1
        self.current_tick = None
        self.done = None
        self._position = None
        self._position_history = None
        self.save_reward = []
        self.rsi = None
        self.mom = None
        self.adx = None
        self.macd = None
        self.masc_sig = None
        self.cci = None

        self.reset()

    def reset(self):
        self.done = False
        self.current_tick = self.start_tick = 0
        self.stock_owned = 0.
        self.stock_price = self.stock_price_history['Adj Close'][self.current_tick]
        self.cash_in_hand = self.capital
        self.save_position = [0]
        self.save_port_val = [self.capital]
        self.save_cash = [self.capital]
        self.total_reward = [0]
        self.stock_value = [0]
        self.sell_buy = ['hold']
        self._position = Positions.Short
        self._position_history = [self._position]
        self.save_stock_price = [self.stock_price]
        self.rsi = self.stock_price_history['rsi'][self.current_tick]
        self.mom = self.stock_price_history['mom'][self.current_tick]
        self.adx = self.stock_price_history['adx'][self.current_tick]
        self.macd = self.stock_price_history['macd'][self.current_tick]
        self.masc_sig = self.stock_price_history['macd sig'][self.current_tick]
        self.cci = self.stock_price_history['cci'][self.current_tick]
        return self.state_vector()

    def state_vector(self):
        vector = np.empty(self.state_dim)
        vector[0] = self.stock_price
        vector[1] = self.stock_owned
        vector[2] = self.cash_in_hand
        vector[3] = self.rsi
        vector[4] = self.mom
        vector[5] = self.adx
        vector[6] = self.macd
        vector[7] = self.masc_sig
        vector[8] = self.cci
        return vector

    def port_val(self):
        return (self.stock_owned * self.stock_price) + self.cash_in_hand

    def trade(self, action):

        if (action == Actions.Sell.value and self._position == Positions.Long):
            # sell
            self.cash_in_hand += ((self.stock_owned * self.stock_price) * (1 - self.trade_fee_bid_percent))
            self.stock_owned -= self.stock_owned
            self.sell_buy.append('sell')

        elif (action == Actions.Buy.value and self._position == Positions.Short):
            # buy
            self.stock_owned += ((self.cash_in_hand * (1 - self.trade_fee_ask_percent)) / self.stock_price)
            self.cash_in_hand -= ((self.cash_in_hand / self.stock_price) * self.stock_price)
            self.sell_buy.append('buy')

    def step(self, action):

        prev_port_val = self.port_val()
        self.current_tick += 1

        if self.current_tick == self.end_tick:
            self.done = True

        self.stock_price = self.stock_price_history['Adj Close'][self.current_tick]
        self.rsi = self.stock_price_history['rsi'][self.current_tick]
        self.mom = self.stock_price_history['mom'][self.current_tick]
        self.adx = self.stock_price_history['adx'][self.current_tick]
        self.macd = self.stock_price_history['macd'][self.current_tick]
        self.masc_sig = self.stock_price_history['macd sig'][self.current_tick]
        self.cci = self.stock_price_history['cci'][self.current_tick]

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self.trade(action)
            self._position = self._position.opposite()

        else:
            self.sell_buy.append('hold')

        self._position_history.append(self._position)

        current_port_val = self.port_val()
        step_reward = current_port_val - prev_port_val
        self.total_reward.append(step_reward)
        self.save_reward.append(step_reward)
        self.save_cash.append(self.cash_in_hand)
        self.save_port_val.append(self.port_val())
        self.save_position.append(self.stock_owned)
        self.stock_value.append(self.stock_owned * self.stock_price)
        self.save_stock_price.append(self.stock_price)

        info = {'stock_price': self.save_stock_price,
                'portfolio_value': self.save_port_val,
                'current_val': self.total_reward,
                'stock_owned': self.save_position,
                'cash_in_hand': self.save_cash,
                'stock_value': self.stock_value,
                'sell_buy': self.sell_buy,
                }
        return self.state_vector(), step_reward, self.done, info

    def get_scaler(self):
        states = []
        self.reset()
        for i in range(self.n_step):
            action = self.action_space.sample()
            state, reward, done, info = self.step(action)
            states.append(state)
            if done:
                break
        scaler = StandardScaler()
        scaler.fit(states)
        return scaler




