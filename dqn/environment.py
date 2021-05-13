import numpy as np
import pandas as pd


class SingleStockEnv:
    '''
    Single Stock for trading

    '''

    def __init__(self, data, capital, pos=200):
        self.stock_price_history = data
        self.n_step = self.stock_price_history.shape[0]
        self.capital = capital
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.state_dim = 9  # [stock price, stock_owned, cash_in_hand, rsi, mom, adx, macd, macd_sig, cci]
        self.save_position = []
        self.save_port_val = []
        self.save_cash = []
        self.stock_value = []
        self.action_space = [0, 1, 2]
        self.sell_buy = None
        self.save_stock_price = []
        self.rsi = None
        self.mom = None
        self.adx = None
        self.macd = None
        self.masc_sig = None
        self.cci = None
        self.save_reward = None
        self.pos = pos
        self.reset()

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

    def reset(self):
        self.current_step = 0
        self.stock_owned = 0
        self.stock_price = self.stock_price_history['Adj Close'][self.current_step]
        self.cash_in_hand = self.capital
        self.save_position = [0]
        self.save_port_val = [self.capital]
        self.save_cash = [self.capital]
        self.stock_value = [0]
        self.sell_buy = ['hold']
        self.save_stock_price = [self.stock_price]
        self.rsi = self.stock_price_history['rsi'][self.current_step]
        self.mom = self.stock_price_history['mom'][self.current_step]
        self.adx = self.stock_price_history['adx'][self.current_step]
        self.macd = self.stock_price_history['macd'][self.current_step]
        self.masc_sig = self.stock_price_history['macd sig'][self.current_step]
        self.cci = self.stock_price_history['cci'][self.current_step]
        self.save_reward = [0]
        return self.state_vector()

    def port_val(self):
        return (self.stock_owned * self.stock_price) + self.cash_in_hand

    def trade(self, action):

        assert action in self.action_space

        if (action == 0) and (self.stock_owned >= self.pos):
            # sell
            self.cash_in_hand += (self.pos * self.stock_price)
            self.stock_owned -= self.pos
            # print('sell')
            self.sell_buy.append('sell')

        elif (action == 1) and (self.cash_in_hand >= (self.pos * self.stock_price)):
            # buy
            self.cash_in_hand -= (self.stock_price * self.pos)
            self.stock_owned += self.pos
            # print('buy')
            self.sell_buy.append('buy')

        else:
            # print('hold')
            self.sell_buy.append('hold')

        self.save_cash.append(self.cash_in_hand)
        self.save_port_val.append(self.port_val())
        self.save_position.append(self.stock_owned)
        self.stock_value.append(self.stock_owned * self.stock_price)
        self.save_stock_price.append(self.stock_price)

    def step(self, action):

        assert action in range(len(self.action_space))

        prev_val = self.port_val()
        self.current_step += 1
        self.stock_price = self.stock_price_history['Adj Close'][self.current_step]
        self.rsi = self.stock_price_history['rsi'][self.current_step]
        self.mom = self.stock_price_history['mom'][self.current_step]
        self.adx = self.stock_price_history['adx'][self.current_step]
        self.macd = self.stock_price_history['macd'][self.current_step]
        self.masc_sig = self.stock_price_history['macd sig'][self.current_step]
        self.cci = self.stock_price_history['cci'][self.current_step]
        self.trade(action)
        current_val = self.port_val()
        reward = current_val - prev_val
        self.save_reward.append(reward)
        done = self.current_step == self.n_step - 1
        info = {'stock_price': self.save_stock_price,
                'portfolio_value': self.save_port_val,
                'current_val': current_val,
                'stock_owned': self.save_position,
                'cash_in_hand': self.save_cash,
                'stock_value': self.stock_value,
                'sell_buy': self.sell_buy
                }
        return self.state_vector(), reward, done, info
