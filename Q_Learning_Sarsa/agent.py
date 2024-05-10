import random
from collections import defaultdict

import math 
import numpy as np 


class Agent:
    """
    Q-Learning Agent
    Args:
        - alpha (learning rate)
        - discount (discount rate)
        - get_legal_actions: function with input state, return the possible actions
    """
    def __init__(self, alpha, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions 
        # use default dict to handle discritized environment case
        self._qvalue = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha 
        self.discount = discount 

    def get_value(self, state):
        pass 

    def get_qvalue(self, state, action):
        return self._qvalue[state][action]
    
    def update_qvalue(self, state, action, new_q_value):
        self._qvalue[state][action] = new_q_value

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        action = None 
        if len(possible_actions) == 0:
            return action 
        q_value_dict = {action: self.get_qvalue(state, action) for action in possible_actions}
        action = max(q_value_dict, key=lambda x: q_value_dict[x])
        return action

    def update(self, state, action, reward, next_state):
        target = reward + self.discount * self.get_value(next_state)
        new_q_value = target * self.alpha + (1 - self.alpha) * self.get_qvalue(state, action)
        self.update_qvalue(state, action, new_q_value)

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

    def get_action(self, state):
        # Pick action with given epsilon-greedy policy
        possible_actions = self.get_legal_actions(state)
        action = None 

        if len(possible_actions) == 0:
            return action
        
        # greedy 
        if (np.random.uniform(0,1 ) < self.epsilon):
            action = np.random.choice(possible_actions)
        else:
            action = self.get_best_action(state)
        return action 


class SoftmaxAgent(Agent):
    def __init__(self, alpha, tau, discount, get_legal_actions):
        super().__init__(alpha, discount, get_legal_actions)
        self.tau = tau 

    def stable_softmax(self, x):
        x = x - x.max()
        e_x = x.exp()
        return e_x / e_x.sum()
    
    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        action = None 

        if len(possible_actions) == 0:
            return None 

        q_value_list = np.array([self.get_qvalue(state, action) for action in possible_actions])
        prob = self.stable_softmax(q_value_list / self.tau)
        action = np.random.choice(possible_actions, p=prob)
        return action 