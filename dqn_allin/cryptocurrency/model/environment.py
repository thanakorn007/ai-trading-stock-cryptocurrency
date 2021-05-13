import numpy as np
import pandas as pd
import enum

#
# class SingleStockEnv:
#     '''
#     Single Stock for trading
#
#     '''
#
#     def __init__(self, data, capital, fee=0.00005):
#         self.stock_price_history = data
#         self.n_step = self.stock_price_history.shape[0]
#         self.capital = capital
#         self.current_step = None
#         self.stock_owned = None
#         self.stock_price = None
#         self.cash_in_hand = None
#         self.state_dim = 9  # [stock price, stock_owned, cash_in_hand, rsi, mom, adx, macd, macd_sig, cci]
#         self.save_position = []
#         self.save_port_val = []
#         self.save_cash = []
#         self.stock_value = []
#         self.action_space = [0, 1, 2]
#         self.sell_buy = None
#         self.save_stock_price = []
#         self.rsi = None
#         self.mom = None
#         self.adx = None
#         self.macd = None
#         self.masc_sig = None
#         self.cci = None
#         self.save_reward = []
#         self.fee = fee
#         self.reset()
#
#     def state_vector(self):
#         vector = np.empty(self.state_dim)
#         vector[0] = self.stock_price
#         vector[1] = self.stock_owned
#         vector[2] = self.cash_in_hand
#         vector[3] = self.rsi
#         vector[4] = self.mom
#         vector[5] = self.adx
#         vector[6] = self.macd
#         vector[7] = self.masc_sig
#         vector[8] = self.cci
#         return vector
#
#     def reset(self):
#         self.current_step = 0
#         self.stock_owned = 0
#         self.stock_price = self.stock_price_history['Adj Close'][self.current_step]
#         self.cash_in_hand = self.capital
#         self.save_position = [0]
#         self.save_port_val = [self.capital]
#         self.save_cash = [self.capital]
#         self.stock_value = [0]
#         self.sell_buy = ['hold']
#         self.save_stock_price = [self.stock_price]
#         self.rsi = self.stock_price_history['rsi'][self.current_step]
#         self.mom = self.stock_price_history['mom'][self.current_step]
#         self.adx = self.stock_price_history['adx'][self.current_step]
#         self.macd = self.stock_price_history['macd'][self.current_step]
#         self.masc_sig = self.stock_price_history['macd sig'][self.current_step]
#         self.cci = self.stock_price_history['cci'][self.current_step]
#         return self.state_vector()
#
#     def port_val(self):
#         return (self.stock_owned * self.stock_price) + self.cash_in_hand
#
#     def trade(self, action):
#
#         assert action in self.action_space
#
#         if (action == 0) and (self.stock_owned > 0):
#             # sell
#             self.cash_in_hand += ((self.stock_owned * self.stock_price) * (1 - self.fee))
#             self.stock_owned -= self.stock_owned
#             self.sell_buy.append('sell')
#
#         elif (action == 1) and (self.cash_in_hand > 0) and (self.stock_owned == 0):
#             # buy
#             self.stock_owned += ((self.cash_in_hand * (1 - self.fee)) / self.stock_price)
#             self.cash_in_hand -= ((self.cash_in_hand / self.stock_price) * self.stock_price)
#             self.sell_buy.append('buy')
#
#         else:
#             # print('hold')
#             self.sell_buy.append('hold')
#
#         self.save_cash.append(self.cash_in_hand)
#         self.save_port_val.append(self.port_val())
#         self.save_position.append(self.stock_owned)
#         self.stock_value.append(self.stock_owned * self.stock_price)
#         self.save_stock_price.append(self.stock_price)
#
#     def step(self, action):
#
#         assert action in self.action_space
#
#         prev_val = self.port_val()
#         self.current_step += 1
#         self.stock_price = self.stock_price_history['Adj Close'][self.current_step]
#         self.rsi = self.stock_price_history['rsi'][self.current_step]
#         self.mom = self.stock_price_history['mom'][self.current_step]
#         self.adx = self.stock_price_history['adx'][self.current_step]
#         self.macd = self.stock_price_history['macd'][self.current_step]
#         self.masc_sig = self.stock_price_history['macd sig'][self.current_step]
#         self.cci = self.stock_price_history['cci'][self.current_step]
#         self.trade(action)
#         current_val = self.port_val()
#         reward = current_val - prev_val
#         self.save_reward.append(reward)
#         done = self.current_step == self.n_step - 1
#         info = {'stock_price': self.save_stock_price,
#                 'portfolio_value': self.save_port_val,
#                 'current_val': current_val,
#                 'stock_owned': self.save_position,
#                 'cash_in_hand': self.save_cash,
#                 'stock_value': self.stock_value,
#                 'sell_buy': self.sell_buy,
#                 }
#         return self.state_vector(), reward, done, info


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
