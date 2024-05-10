from agent import *

import argparse
import random 
import math

import numpy as np 
import matplotlib.pyplot as plt
import gym

class QLearningAgent(EpsilonGreedyAgent):

    def get_value(self, state):
        best_action = self.get_best_action(state)
        if best_action is not None:
            return self.get_qvalue(state, best_action)
        return None 
    

def train_one_epoch(env, agent, t_max=10**4):
    s, _ = env.reset()
    total_reward = 0.0
    for t in range(t_max):
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)
        agent.update(s, a, r, next_s) 
        s = next_s 
        total_reward += r
        if done:
            break 
    return total_reward

def main(env, args):
    np.random.seed(args.seed)
    n_actions = env.action_space.n
    get_legal_actions = lambda s: range(n_actions)
    agent = QLearningAgent(args.alpha, args.epsilon, args.gamma, get_legal_actions)
    rewards = []
    for i in range(1000):
        rewards.append(train_one_epoch(env, agent))
        # reduce epsilon overtime
        agent.epsilon *= 0.99 
        if i % 10 == 0:
            print(f"Episode {i}: epsilon = {agent.epsilon}, reward = {np.mean(rewards[-10:])}")
    
    plt.plot(rewards)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    # For these and any other arguments you add, ReCodEx will keep your default value.
    parser.add_argument("--alpha", default=..., type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=..., type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
    args = parser.parse_args()
    env = gym.make("Taxi-v3")
    main(env, args)