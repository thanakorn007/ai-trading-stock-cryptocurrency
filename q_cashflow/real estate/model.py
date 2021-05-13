import numpy as np
from sklearn.preprocessing import StandardScaler










class LinearModel:
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, s):
        return s.dot(self.W) + self.b

    def sgd(self, s, Y, learning_rate=0.001, momentum=0.9):
        num_values = np.prod(Y.shape)
        Y_hat = self.predict(s)
        Y_hat = self.predict(s)  # shape (1, 8)
        gW = 2 * s.T.dot(Y_hat - Y) / num_values  # vector shape (7, 8)
        gb = 2 * (Y_hat - Y).sum(axis=0) / num_values  # scalar

        self.vW = momentum * self.vW - learning_rate * gW  # vector shape (7, 8)
        self.vb = momentum * self.vb - learning_rate * gb  # scalar

        self.W += self.vW  # vector shape (7, 8)
        self.b += self.vb  # scalar

        mse = np.mean((Y_hat - Y) ** 2)
        self.losses.append(mse)

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class StockEnv:
    def __init__(self, data, capital, trade_fee_bid_percent=0.01, trade_fee_ask_percent=0.005):
        self.trade_fee_bid_percent = trade_fee_bid_percent / 100  # percent
        self.trade_fee_ask_percent = trade_fee_ask_percent / 100  # percent

        self.stock_price_history = data['Adj Close']
        self.n_step = self.stock_price_history.shape[0]
        self.capital = capital
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.state_dim = 5  # [stock price, stock_owned, cash__in_hand, rsi, mom]
        self.save_position = []
        self.save_port_val = []
        self.save_cash = []
        self.stock_value = []
        self.action_space = [0, 1]
        self.sell_buy = []
        self.save_stock_price = []
        self.rsi = data['rsi']
        self.mom = data['mom']
        self.reset()

    def state_vector(self):
        vector = np.empty(self.state_dim)
        vector[0] = self.stock_price
        vector[1] = self.stock_owned
        vector[2] = self.cash_in_hand
        vector[3] = self.rsi[self.current_step]
        vector[4] = self.mom[self.current_step]
        return vector

    def reset(self):
        self.current_step = 0
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.current_step]
        self.cash_in_hand = self.capital
        self.save_position = [0]
        self.save_port_val = [self.capital]
        self.save_cash = [self.capital]
        self.stock_value = [0]
        self.sell_buy = ['hold']
        self.save_stock_price = [self.stock_price]
        return self.state_vector()

    def port_val(self):
        return (self.stock_owned * self.stock_price) + self.cash_in_hand

    def cal_position(self):
        high = self.stock_price_history.max()
        mid_price = high / 2
        max_pos = self.capital / mid_price
        slope = max_pos / -high
        c = -slope * high

        return slope * self.stock_price + c, high, max_pos

    def trade(self, action):
        '''
        action 0 = hold
        action 1 = take action
        '''
        assert action in self.action_space

        if action == 1:
            position_sh, high, max_pos = self.cal_position()
            diff = position_sh - self.stock_owned
            if diff < 0:
                # sell
                self.stock_owned = position_sh
                self.cash_in_hand += (abs(diff) * self.stock_price * (1 - self.trade_fee_ask_percent))
                self.sell_buy.append('sell')


            elif diff > 0:
                # buy
                self.stock_owned = position_sh
                self.cash_in_hand -= (abs(diff) * self.stock_price * (1 + self.trade_fee_bid_percent))
                self.sell_buy.append('buy')

            else:
                self.sell_buy.append('hold')

        else:
            self.sell_buy.append('hold')

        self.save_cash.append(self.cash_in_hand)
        self.save_port_val.append(self.port_val())
        self.save_position.append(self.stock_owned)
        self.stock_value.append(self.stock_owned * self.stock_price)
        self.save_stock_price.append(self.stock_price)

    def step(self, action):
        assert action in self.action_space

        prev_val = self.port_val()
        self.current_step += 1
        self.stock_price = self.stock_price_history[self.current_step]

        self.trade(action)
        current_val = self.port_val()
        reward = current_val - prev_val
        done = self.current_step == self.n_step - 1
        info = {'stock_price': self.save_stock_price,
                'portfolio_value': self.save_port_val,
                #                 'current_val': self.port_val(),
                'stock_owned': self.save_position,
                'cash_in_hand': self.save_cash,
                'stock_value': self.stock_value,
                'sell_buy': self.sell_buy
                }

        return self.state_vector(), reward, done, info


def get_scaler(env):
    states = []
    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def play_one_episode(agent, env, is_train, scaler):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)

        state = next_state

    return info