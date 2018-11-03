# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register
import sys, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as pr

    
if __name__ == '__main__':
       
    env = gym.make("CartPole-v0")
    
    # Input and output size based on the Env
    input_size = env.observation_space.shape[0]     # 4
    output_size = env.action_space.n                # 2
    learning_rate = 1e-1
    
    # These lines establish the feed-forward part of the network used to choose actions
    X = tf.placeholder(shape=[None,input_size], dtype=tf.float32, name="input_x") # state input
   
    # First layer of weights
    W1 = tf.get_variable("W1", shape=[input_size, output_size],
                         initializer=tf.contrib.layers.xavier_initializer())    # weight  # (4, 2)
    print(W1)    # (4, 2)

    Qpred = tf.matmul(X, W1) # Out Q prediction
 
    # We need to define the parts of the network needed for learning a policy
    Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)    # Y label

    # Cost function
    loss = tf.reduce_sum(tf.square(Y-Qpred))
    # Learning
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Set Q-learning related parameters
    gamma = 0.9  # discount factor
    num_episodes = 200
    
    # create lists to contain total rewards and steps per episode
    rList = []   
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for episode in range(num_episodes):
            # Reset environment and get first new observation
            state = env.reset()
            e = 1. / ((episode/10)+1)  # decaying E-greedy
            rAll = 0
            step_count = 0
            terminal = False
            local_loss = []
                  
            # The Q-Network training
            while not terminal:
                step_count += 1
                x = np.reshape(state, [1, input_size])
                # Choose an action by greedily (with e chance of random action) from the Q-network
                Qs = sess.run(Qpred, feed_dict={X: x})
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Qs)
                   
                # Get new state and reward from environment
                new_state, reward, terminal, _ = env.step(action)
                if terminal:
                    # Update Q, and no Qs+1, since it's a terminal state
                    Qs[0,action] = -100
                else:
                    x1 = np.reshape(new_state, [1, input_size])
                    # Obtain the Q_sq values by feeding the new state through our network
                    Qs1 = sess.run(Qpred, feed_dict={X: x1})
                    # Update Q
                    Qs[0,action] = reward + gamma * np.max(Qs1)
                    #print(Qs)
                    
                # Train our network using target (Y) and predicted Q (Qpred) values
                sess.run(train, feed_dict={X: x, Y: Qs})
                                    
                rAll += reward
                state = new_state
                
            rList.append(step_count)
            print("Episode: {} steps: {}".format(episode, step_count))
            # If last 10's avg steps are 500, it's good enough
            if len(rList) > 10 and np.mean(rList[-10:]) > 500:
                break
        
        print("Success rate: "+ str(sum(rList)/num_episodes))
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
        
        
        # see our trained network in action
        observation = env.reset()
        reward_sum = 0
        while True:
            env.render()
            
            x = np.reshape(observation, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            action = np.argmax(Qs)
            
            observation, reward, terminal, _ = env.step(action)
            reward_sum += reward
            if terminal:
                print("Total score: {}".format(reward_sum))
                break
    
