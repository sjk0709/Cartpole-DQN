# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register
from gym import wrappers
import sys, os
import tensorflow as tf
import numpy as np
#import matplotlib as mpl
#mpl.use('TkAgg')

 
def main():
    #    env = gym.make('CartPole-v0')

    max_episode_steps = 500
    
    register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
        reward_threshold=475.0,
    )
    env = gym.make('CartPole-v2')

    # Constants defining our neural network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print('input_size : ', input_size)
    print('output_size : ', output_size)

    #Now, save the graph
    checkpoint_path = "CartPole0/"
    checkpoint_state_name = "checkpoint_state"    
    
 
    with tf.Session() as sess:
        
        checkpointState = tf.train.get_checkpoint_state(checkpoint_path, latest_filename=checkpoint_state_name)   
        print('checkpoint :', checkpointState)
        
        sess=tf.Session()    
        
        tf.global_variables_initializer().run()
        
       
#        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
       # targetDQN = dqn.DQN(sess, input_size, output_size, name="target")   

        print("========================================================================")
        if checkpointState and checkpointState.model_checkpoint_path: 
            #First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph(checkpoint_path+'network-0.meta')
            saver.restore(sess, checkpointState.model_checkpoint_path)
            print("Loaded checkpoints from %s" % checkpointState.model_checkpoint_path)
        elif False:
            raise Exception("Could not load checkpoints for playback")
        else:
            print("TTTTTTTTT")
        print("=========================================================================")
        
        graph = tf.get_default_graph()
        
        X = graph.get_tensor_by_name("main/input_x:0")
        Q = graph.get_tensor_by_name("main/Q:0")   

        # Game Test     
        state = env.reset()
        reward_sum = 0        
        stepCount = 0
        while True:
            env.render()            
            x = np.reshape(state, [1, input_size])           
            action = np.argmax(sess.run(Q, feed_dict={X: x}))
            state, reward, terminal, _ = env.step(action)
            reward_sum += reward
            stepCount += 1
            if terminal:
                print("You have to train the model")
                print("Last Step: {}".format(stepCount), "Total score: {}".format(reward_sum))
                break
            elif stepCount>=max_episode_steps:
                print("Very Good!!!!!!!!!")
                print("Last Step: {}".format(stepCount), "Total score: {}".format(reward_sum))
                break
             
#        bot_play(Q)

if __name__ == "__main__":
    main()