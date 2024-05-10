import numpy as np 

import gym 
from gym.core import ObservationWrapper


class Discretizer(ObservationWrapper):
    def observation(self, state):
        max_state = np.array([0.75, 2, 0.3, 3])
        min_state = - max_state
        state = np.clip(state, min_state, max_state)
        state[0] = np.round(state[0], 2)
        state[1] = np.round(state[1], 1)
        state[2] = np.round(state[2], 2)
        state[3] = np.round(state[3], 1)

        return tuple(state)
    

def make_env():
    env = gym.make("CartPole-v0")
    env = Discretizer(env)
    return env