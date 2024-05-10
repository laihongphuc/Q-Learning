from agent import *

import argparse
import random 
import math

import numpy as np 
import matplotlib.pyplot as plt
import gym
import wandb 

class ExpGreedySarsaAgent(EpsilonGreedyAgent):
    """
    Greedy Policy is (1-epsilon) with best action, and epsilon with random action 
    """
    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None 
        elif len(possible_actions) == 1:
            return self.get_qvalue(state, possible_actions[0])
        best_action = self.get_best_action(state)
        other_action_value = 0
        for action in possible_actions:
            if action == best_action:
                continue
            other_action_value += self.get_qvalue(state, action)
        random_action = np.random.choice(possible_actions)
        value = (1 - self.epsilon) * self.get_qvalue(state, best_action) + self.epsilon / (len(possible_actions) - 1) * self.get_qvalue(state, random_action)
        return value
    

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
    agent = ExpGreedySarsaAgent(args.alpha, args.epsilon, args.gamma, get_legal_actions)
    rewards = []
    for i in range(1000):
        reward = train_one_epoch(env, agent)
        wandb.log({"Taxi-v3/Reward": reward})
        rewards.append(reward)
        # reduce epsilon overtime
        agent.epsilon *= 0.99 
        if i % 10 == 0:
            print(f"Episode {i}: epsilon = {agent.epsilon}, reward = {np.mean(rewards[-10:])}")
    



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
    wandb.finish()
    wandb.init(project="Q-Learning", config=args)
    wandb.run.name = f'run_expected_sarsa_greedy_policy_alpha={args.alpha}_epsilon={args.epsilon}_gamma={args.gamma}'
    main(env, args)
    wandb.finish()