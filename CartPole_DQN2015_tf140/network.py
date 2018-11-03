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

    
class DQN:
    
    def __init__(self, session, input_size, output_size, name="policy"):
        
        self.sess = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        
        self.build_network()
        
    def build_network(self, h_size=16, learning_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self.state = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name="state")
            self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action" )
            
            dense1 = tf.layers.dense(inputs=self.state, units=h_size, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=self.kernel_regularizer)
            
            dense2 = tf.layers.dense(inputs=dense1, units=h_size, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=self.kernel_regularizer)
            
#            dense3 = tf.layers.dense(inputs=dense2, units=h_size, activation=tf.nn.relu,
#                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                    kernel_regularizer=self.kernel_regularizer)
            
            self.output = tf.layers.dense(inputs=dense2, units=self.output_size, 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=self.kernel_regularizer)
            
#            # First layer of weights
#            W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
#            b1 = tf.Variable(tf.constant(0.1, shape=[h_size]))
#            layer1 = tf.nn.tanh(tf.matmul(self._X, W1)+b1)
#            
#            # Second layer of weights
#            W2 = tf.get_variable("W2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
#            b2 = tf.Variable(tf.constant(0.1, shape=[h_size]))
#            layer2 = tf.nn.relu(tf.matmul(layer1, W2)+b2)
#            
#            W3 = tf.get_variable("W3", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
#            b3 = tf.Variable(tf.constant(0.1, shape=[self.output_size]))
#            # Q prediction
#            self._Qpred = tf.matmul(layer2, W3, name="Q")+b3
          
            self.one_hot = tf.one_hot(self.action, self.output_size)
#            print(self.one_hot)
            self.Q = tf.reduce_sum(self.output*self.one_hot , axis=1)
            
            self.prob = tf.nn.softmax(self.output, name="prob")
        # we need to define the parts of the network needed for learning a
        
        # policy
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(self.Y - self.Q))
        
        # Learning
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
    def predict(self, state):
        state = np.reshape(state, [-1, self.input_size])
        return self.sess.run(self.output, feed_dict={self.state: state})
    
    def update(self, state, action, y):
        return self.sess.run([self.loss, self.train], feed_dict={self.state: state, self.action: action, self.Y: y})
    
