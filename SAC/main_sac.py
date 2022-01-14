# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 00:59:21 2022

@author: dltns
"""

import os
import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils1 import plot_learning_curve
from gym import wrappers
import matplotlib.pyplot as plt



if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    print(env.action_space)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
        n_actions=env.action_space.shape[0])
    n_games = 20
    filename_1 = 'inverted_pendulum_avg.png'
    filename_2 = 'inverted pendulum.png'

    figure_file_1 = 'plots/' + filename_1
    figure_file_2 = 'plots/' + filename_2

    best_score = env.reward_range[0]
    score_history = []
    avg_score_save=[]
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score=np.mean(score_history[-20:])
        avg_score_save.append(avg_score)

        if avg_score > best_score:
            if len(score_history)>1:
                if score_history[-1] > score_history[-2]:
                    best_score = avg_score
                    if not load_checkpoint:
                        agent.save_models()
            else :
                best_score = avg_score
                if not load_checkpoint:
                        agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, avg_score_save, figure_file_1)
        plot_learning_curve(x, score_history, figure_file_2)
        np.savetxt('./data/bullet_v0_data_sac.csv', avg_score_save)
        plt.legend(['Running average of previous 20 scores','scores for each episode'])
        plt.title(['InvertedPendulumBulletEnv-v0'])


