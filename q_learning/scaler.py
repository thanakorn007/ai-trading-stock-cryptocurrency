from sklearn.preprocessing import StandardScaler
import numpy as np

def get_scaler(env):
    states = []
    for i in range(env.n_step):
        action = np.random.choice(range(len(env.action_space)))
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler
