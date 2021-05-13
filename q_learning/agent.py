from linear_model import LinearModel
import numpy as np


class QAgent(object):
    def __init__(self, state_size, action_size, discount_factor=0.95, ticker=''):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tichker = ticker
        self.model = LinearModel(state_size, action_size)
        self.save_epsilon = []

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        act_values = self.model.forward(state)
        return np.argmax(act_values[0])  # returns action


    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.forward(next_state), axis=1)

        target_full = self.model.forward(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.save_epsilon.append(self.epsilon)



    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)


