import numpy as np




class LinearModel:
    def __init__(self, input_sz, action_sz):
        self.W = np.random.randn(input_sz, action_sz) / np.sqrt(input_sz)
        self.b = np.zeros(action_sz)
        self.vW = 0
        self.vb = 0
        self.losses = []

    def forward(self, state):
        return state.dot(self.W) + self.b

    def sgd(self, state, target, lr=0.001, momentum=0.9):
        num_values = np.prod(target.shape)
        output = self.forward(state)

        gW = 2 * state.T.dot(target - output) / num_values
        gb = 2 * (target - output).sum(axis=0) / num_values

        self.vW = momentum * self.vW - lr * gW
        self.vb = momentum * self.vb - lr * gb

        loss = np.mean((target - output) ** 2)
        self.losses.append(loss)

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']