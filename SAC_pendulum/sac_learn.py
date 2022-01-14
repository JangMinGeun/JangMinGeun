# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:01:01 2022

@author: dltns
"""

# SAC learn (tf2 subclassing version)
# coded by St.Watermelon

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

from replaybuffer import ReplayBuffer


# actor network
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]  # std bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        # Scale output to [-action_bound, action_bound]
        mu = Lambda(lambda x: x*self.action_bound)(mu)
        # clipping std
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        log_pdf = normal_prob.log_prob(action)
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)

        return action, log_pdf


# critic network
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')


    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q


class SACagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0003
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.005
        self.ALPHA = 0.5

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        ## create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))

        self.critic = Critic()
        self.target_critic = Critic()

        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []

    ## actor policy
    def get_action(self, state):
        mu, std = self.actor(state)
        action, _ = self.actor.sample_normal(mu, std)
        return action.numpy()[0]


    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)


    ## single gradient update on a single batch data
    def critic_learn(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q-q_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## train the actor network
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            soft_q = self.critic([states, actions])

            loss = tf.reduce_mean(self.ALPHA * log_pdfs - soft_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## computing soft Q target
    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.actor.load_weights(path + '/pendulum_actor.h5')
        self.critic.load_weights(path + '/pendulum_critic.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()

            while not done:
                # visualize the environment
                #self.env.render()
                # pick an action: shape = (1,)
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe reward, new_state
                next_state, reward, done, _ = self.env.step(action)
                # add transition to replay buffer
                train_reward = (reward + 8) / 8

                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:  # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target soft Q-values
                    next_mu, next_std = self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

                    target_qs = self.target_critic([next_states, next_actions])

                    target_qi = target_qs - self.ALPHA * next_log_pdf

                    # compute TD targets
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    # train critic using sampled batch
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # train actor
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))

                    # update both target network
                    self.update_target_network(self.TAU)

                # update current state
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            #print('Now save')
            self.actor.save_weights("save_weights/pendulum_actor.h5")
            self.critic.save_weights("save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()