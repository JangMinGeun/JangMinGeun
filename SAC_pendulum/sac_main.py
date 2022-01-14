# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:02:58 2022

@author: dltns
"""

# SAC main (tf2 subclassing API version)
# coded by St.Watermelon

import gym
from sac_learn import SACagent
import pybullet_envs

def main():

    max_episode_num = 160
    env = gym.make("InvertedPendulumBulletEnv-v0")
    agent = SACagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()