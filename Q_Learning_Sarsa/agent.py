import random
from collections import defaultdict

import math 
import numpy as np 


class Agent:
    """
    Q-Learning Agent
    Instance variables:
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """
    def __init__(self, alpha, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions 
        # use default dict to handle discritized environment case
        self._qvalue = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha 
        self.discount = discount 

    def get_qvalue(self, state, action):
        return self._qvalue[state][action]
    
    def get_best_action(self, state):
        pass 

    def update(self, state, action, reward, next_state):
        pass 

    def get_action(self, state):
        pass 


class EpsilonGreedyAgent(Agent):
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Epsilon greedy agent
        Instance variables:
            - self.epsilon: exploration rate
            - self.alpha
            - self.discount
        """
        super().__init__(alpha, discount, get_legal_actions)
        self.epsilon = epsilon


class SoftmaxAgent(Agent):
    def __init__(self, alpha, tau, discount, get_legal_actions):
        super().__init__(alpha, discount, get_legal_actions)
        self.tau = tau 

    def stable_softmax(self, x):
        x = x - x.max()
        e_x = x.exp()
        return e_x / e_x.sum()