import argparse
import random 
import math 
import sys 
sys.path.append("..")

import numpy as np 
import gym 
import matplotlib.pyplot as plt 
import wandb

from q_learning import QLearningAgent, train_one_epoch
from exp_sarsa_greedy_policy import ExpGreedySarsaAgent
from exp_sarsa_softmax_policy import ExpSoftmaxSarsaAgent


def draw_policy(agent):
    """ Prints CliffWalkingEnv policy with arrows. Hard-coded. """

    env = gym.make('CliffWalking-v0', render_mode='ansi')
    env.reset()
    grid = [x.split('  ') for x in env.render().split('\n')[:4]]

    n_rows, n_cols = 4, 12
    start_state_index = 36
    actions = '^>v<'

    for yi in range(n_rows):
        for xi in range(n_cols):
            if grid[yi][xi] == 'C':
                print(" C ", end='')
            elif (yi * n_cols + xi) == start_state_index:
                print(" X ", end='')
            elif (yi * n_cols + xi) == n_rows * n_cols - 1:
                print(" T ", end='')
            else:
                print(" %s " %
                      actions[agent.get_best_action(yi * n_cols + xi)], end='')
        print()


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
    sarsa_greedy_agent = ExpGreedySarsaAgent(args.alpha, args.epsilon, args.gamma, get_legal_actions)
    sarsa_softmax_agent = ExpSoftmaxSarsaAgent(args.alpha, args.tau, args.gamma, get_legal_actions)
    q_agent = QLearningAgent(args.alpha, args.epsilon, args.gamma, get_legal_actions)
    reward_sarsa_greedy, reward_sarsa_softmax, reward_q_agent = [], [], []
    for i in range(5000):
        reward_1 = train_one_epoch(env, sarsa_greedy_agent)
        reward_2 = train_one_epoch(env, sarsa_softmax_agent)
        reward_3 = train_one_epoch(env, q_agent)
        # wandb.log({"CliffWalking/Reward/Sarsa_Greedy": reward_1})
        # wandb.log({"CliffWalking/Reward/Sarsa_Softmax": reward_2})
        # wandb.log({"CliffWalking/Reward/Q_Learning": reward_3})
        reward_sarsa_greedy.append(reward_1)
        reward_sarsa_softmax.append(reward_2)
        reward_q_agent.append(reward_3)
        # reduce epsilon overtime
        q_agent.epsilon *= 0.99 
        sarsa_greedy_agent.epsilon *= 0.99
        if i % 10 == 0:
            print(f"Episode {i}: epsilon = {sarsa_greedy_agent.epsilon}, reward = {np.mean(reward_sarsa_greedy[-100:])}")
            print(f"Episode {i}: reward = {np.mean(reward_sarsa_softmax[-100:])}")
            print(f"Episode {i}: epsilon = {q_agent.epsilon}, reward = {np.mean(reward_q_agent[-100:])}")
    print("Q-Learning")
    draw_policy(q_agent)
    print("Sarsa-Greedy")
    draw_policy(sarsa_greedy_agent)
    print("Sarsa-Softmax")
    draw_policy(sarsa_softmax_agent)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    # For these and any other arguments you add, ReCodEx will keep your default value.
    parser.add_argument("--alpha", default=..., type=float, help="Learning rate.")
    parser.add_argument("--tau", default=..., type=float, help="Softmax temperature.")
    parser.add_argument("--epsilon", default=..., type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
    args = parser.parse_args()
    env = gym.make("CliffWalking-v0")
    # wandb.finish()
    # wandb.init(project="Q-Learning", config=args)
    # wandb.run.name = f'compare_q_learning_sarsa_cliffwalking_alpha={args.alpha}_epsilon={args.epsilon}_gamma={args.gamma}_tau={args.tau}'
    main(env, args)
    # wandb.finish()