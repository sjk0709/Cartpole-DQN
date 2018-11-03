# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register
from gym import wrappers

import sys, os, time

import numpy as np
#import matplotlib as mpl
#mpl.use('TkAgg')
import random 
from collections import deque
from collections import namedtuple

import tensorflow as tf
#from xml.etree.ElementTree import Element, SubElement, dump, ElementTree
import network
   

def createFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
class DQN2015    :
        
    def __init__(self, game, settings):
        self.env = game
        self.settings = settings
        
        # Constants defining our neural network
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        print('input_size : ', self.input_size)  # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        print('output_size : ', self.output_size)  # Left, Right
        
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal'))
        
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0
        
        model_folder_name = "models/"
        createFolder(model_folder_name)    
            
        self.save_folder_path = model_folder_name + self.settings.save_folder_file[0]
        createFolder(self.save_folder_path)
        
        self.checkpoint_state = "checkpoint_state"
        self.save_model_path = self.save_folder_path + self.settings.save_folder_file[1]
        self.optimal_model_path = self.save_folder_path + "optimal"
        
        
        self.load_folder_path = model_folder_name + self.settings.load_folder_file[0]
        self.load_model_path = self.load_folder_path + self.settings.load_folder_file[1] + ".meta"
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True) )
        self.sess = tf.Session()
          
        
        # declare model
        self.policyNet = network.DQN(self.sess, self.input_size, self.output_size, name="policy")
        self.targetNet = network.DQN(self.sess, self.input_size, self.output_size, name="target")
             
#        if 'session' in locals() and self.sess is not None:
#            print('Close interactive session')
#            session.close()
        
        self.saver = tf.train.Saver()        
        checkpoint = tf.train.get_checkpoint_state(self.load_folder_path, latest_filename=self.checkpoint_state)
        
        self.sess.run(tf.global_variables_initializer())
        
        
        if checkpoint and checkpoint.model_checkpoint_path:
            print(checkpoint)
            print(checkpoint.model_checkpoint_path)
#            self.saver = tf.train.import_meta_graph(self.load_model_path)
#            self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("%s has been loaded." % checkpoint.model_checkpoint_path)
        
        else :
            print("First learning.")
        
        
        

            
    def get_copy_var_ops(self, *, dest_scope_name="target", src_scope_name="policy"):
        # Copy variables src_scope to dest_scope
        op_holder = []
        
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
        
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))
            
        return op_holder
        
    def replay_train(self, policy_net, target_net, train_batch, gamma=0.9):
        
        batch = self.transition(*zip(*train_batch))        
        
        # Compute max_a'Q(S_{t+1}, a', theta-) for all next states.
        max_Qtarget_next_s = np.max( self.targetNet.predict(batch.next_state), axis=1 )
        
#        print(max_Qtarget_next_s)
        
        # R_{t+1} + gamma * max_a'Q(S_{t+1}, a', theta-)
        y = batch.reward + gamma * max_Qtarget_next_s * batch.terminal
       

        # Q(S_t, A_t, theta)               
        # Train our network using target and predicted Q values on each episode
        return self.policyNet.update(batch.state, batch.action, y)

    def select_action(self, state):
        
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
            
        self.steps_done += 1
        
        if random.random() > self.eps_threshold:            
            return np.argmax(self.policyNet.predict(state))
        else:
            return self.env.action_space.sample()
        


    def train(self, max_episodes):
        # Save our model
#        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)
        
        # store the previous observations in replay memory
        replay_buffer = deque()
        start_time = time.perf_counter()
        
        current_max_step = self.settings.initialStepForOptimalModel
                
        # initial copy q_net -> target_net
        copy_ops = self.get_copy_var_ops( dest_scope_name="target",
                                    src_scope_name="policy")
        self.sess.run(copy_ops)
        
        
        # train my model
        for episode in range(max_episodes):
#            self._e = 1. / ((episode / 100) + 1)
            terminal = False
            step_count = 0
            loss = 0.0
            state = self.env.reset()
            
            while not terminal:
                
                action = self.select_action(state)
                
                # Get new state and reward from environment
                next_state, reward, terminal, _ = self.env.step(action)
                
                terminalNo = 1.0
                if terminal:    # big penalty   
                    terminalNo = 0.0
#                    reward = -100                   
                    
                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, terminalNo))
                if len(replay_buffer) > self.settings.replayMemory:
                    replay_buffer.popleft()
                    
                state = next_state
                step_count += 1
            
            
            
            if step_count > current_max_step:
                current_max_step = step_count
                # save model 
                save_path = self.saver.save(self.sess, self.save_model_path, global_step=0, latest_filename=self.checkpoint_state)
                save_path = self.saver.save(self.sess, self.optimal_model_path)
#                f = open(self.checkpoint_dir +"parameters.txt", 'w')
#                f.write(str(self._e))
#                f.close() 
                print('=====================================================================')
                print("Episode: {}  steps: {} <= Good enough!!!!!!!!!!".format(episode, step_count))
                print('Current checkpoint has been saved')
                print('=====================================================================')
                      
            else :
                print("Episode: {}  steps: {}".format(episode, step_count))

#            if step_count > 10000:
#                pass
            
            if episode % 10 == 0 and len(replay_buffer)>self.settings.batch_size:
                # Get a random batch of experiences.
                for _ in range(50):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, self.settings.batch_size)
                    loss, _ = self.replay_train(self.policyNet, self.targetNet, minibatch)
                
                self.sess.run(copy_ops)  # main -> target after training
                
                print('=====================================================================')   
                print("Eps threshold :", self.eps_threshold)
                print("Loss :", loss)
                print('=====================================================================')

#            if self._e > self._FINAL_RANDOM_ACTION_PROB and len(self._replay_buffer) > self._OBSERVATION_STEPS:
#                self._e -= (self._INITIAL_RANDOM_ACTION_PROB - self._FINAL_RANDOM_ACTION_PROB) / self._EXPLORE_STEPS
      
                     
        # Save training information and our model                         
#        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)        
         
        elapsed_time = (time.perf_counter() - start_time)
        print('=====================================================================')
        print('Elapsed %.3f seconds.' % elapsed_time)
        print('%.0f h' % (elapsed_time/3600), '%.0f m' % ((elapsed_time%3600)/60) , '%.0f s' % (elapsed_time%60) )
        print('Learning Finished!')   
        print('=====================================================================')
                
        self.test()
        # See our trained bot in action
#        env2 = wrappers.Monitor(env, 'gym-results', force=True)        
#        for i in range(200):
#            self.bot_play(self._mainDQN, env=env2)
            
        
    def bot_play(self, policyNet, use_render=True):
        # See our trained network in action
        state = self._game.reset()
        reward_sum = 0
        step_count = 0
        while True:
            if use_render:
                self.env.render()
                
            action = np.argmax(policyNet.predict(state))
            
            state, reward, terminal, _ = self.env.step(action)
            reward_sum += reward
            step_count += 1            
            if terminal:
#                print("You have to train the model")
#                print("Total score: {}".format(reward_sum))
                break
            
                
        self.env.close()
        return step_count
        
    def test(self, use_render=True):  
                  
#        game2 = wrappers.Monitor(self._game, 'gym-results', force=True)        
#        for i in range(200):            
        self.bot_play(self.policyNet, use_render)
     
        
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
        
def main():
    
    print("Tensorflow version :", tf.__version__)


    
    settings = dotdict({
            'training' : True, 
            'isGPU' : True,
            'load_model' : True, 
            'save_folder_file' : ("Cartpole0/", 'checkpoint0'),
            'load_folder_file' : ("Cartpole0/", 'checkpoint0-0'),
            'replayMemory' : 50000,
            'initialStepForOptimalModel' : 5000, 
            'maxEpisodes' : 10000,
            'batch_size' : 128, 
            'learning_rate' : 1e-1,
            'graphName' : "CartPole0",
            })
    
    gym.envs.register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        reward_threshold=9750,
    )
    game = gym.make('CartPole-v2')
#    game.reset()
#    game = gym.make('CartPole-v0')
    
    cartPole = DQN2015(game, settings)
    
    if settings.training:
        cartPole.train(settings.maxEpisodes)
    else:
        cartPole.test(use_render=True)
    

if __name__ == "__main__":
    main()