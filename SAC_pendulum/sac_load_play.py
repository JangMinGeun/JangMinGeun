# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:03:23 2022

@author: dltns
"""

import gym
from sac_learn import SACagent
import tensorflow as tf

def main():

    env = gym.make("Pendulum-v1")
    agent = SACagent(env)

    agent.load_weights('save_weights')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()