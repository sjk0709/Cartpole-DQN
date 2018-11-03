# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register
from gym import wrappers

import random 
from collections import deque
from collections import namedtuple
#from xml.etree.ElementTree import Element, SubElement, dump, ElementTree

import sys, os, time
sys.path.append(os.pardir)  # parent directory
import numpy as np
from xml.etree.ElementTree import Element, SubElement, dump, ElementTree
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt


from network import Network0 as Net

        
 
def weights_init(m):
    classname = m.__class__.__name__
#            print(classname) 
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
#                m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.1)
        print("xavier_uniform")
        
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class DQN2015    :
        
    def __init__(self, game, args):
        self.env = game
        
        # Constants defining our neural network
        self.args = args
        self._input_size = self.env.observation_space.shape[0]
        self._output_size = self.env.action_space.n
        print('input_size : ', self._input_size)  # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        print('output_size : ', self._output_size)  # Left, Right

        self.device = torch.device("cuda" if args.isGPU else "cpu")
        
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal'))
        
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0


        self.modelFilePath = "models/"
        if not os.path.exists(self.modelFilePath):
            os.mkdir(self.modelFilePath)       
                    
        self.saveModelFilePath = self.modelFilePath + self.args.save_folder_file[0]
        if not os.path.exists(self.saveModelFilePath):
            os.mkdir(self.saveModelFilePath)  
        
        self.loadModelFilePath = self.modelFilePath + self.args.load_folder_file[0]
#        self.resultFolderPath = "results/"
#        if not os.path.exists(self.resultFolderPath):
#            os.mkdir(self.resultFolderPath)  
        
        # declare moedl
        self.policyNet = Net( self._input_size, self._output_size ).to(self.device)
        self.targetNet = Net( self._input_size, self._output_size ).to(self.device)
        self.policyNet.apply(weights_init)    
    
        # load moedl and parameters
        self.loadModelPath = self.loadModelFilePath + self.args.load_folder_file[1] + '_all.pkl'
        self.loadParamsPath = self.loadModelFilePath + self.args.load_folder_file[1] + '_params.pkl'
        
        self.saveModelPath = self.saveModelFilePath + self.args.save_folder_file[1] + '_all.pkl'
        self.saveParamsPath = self.saveModelFilePath + self.args.save_folder_file[1] + '_params.pkl'
        
        self.optimalModelSavePath = self.saveModelFilePath + 'optimal_all.pkl'
        self.optimalParamsSavePath = self.saveModelFilePath + 'optimal_params.pkl'
        
        if self.args.load_model:            
#            print("\n--------" + self.policyParamsPath + " is restored.--------\n")
            try:       
                self.policyNet = torch.load(self.loadModelPath, map_location=lambda storage, location: storage) 
                self.targetNet.load_state_dict(self.policyNet.state_dict())
                self.targetNet.eval()
#                self.policyNet.load_state_dict(torch.load(self.loadParamsPath))      # it loads only the model parameters (recommended)                  
#                f = open(self.modelFilePath + "parameters.txt", 'r')
#                self._e = float(f.readline())
#                print("e :", self._e)
#                f.close()                                       
                self.EPS_START = 0.5
                
                print("\n--------" + self.args.load_folder_file[1] + " is restored.--------\n")
#                print("\n All parameters are restored.--------\n")
                
            except:
                print("\n--------There are no models.--------\n")
                print("\n--------First learning.--------\n")
                pass
        else:          
            print("\n--------First learning.--------\n")
      
            
        self.policyNet.to(self.device)
        self.targetNet.to(self.device)

        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)        
        self.loss_func = nn.MSELoss()
#        self.loss_func = nn.CrossEntropyLoss()
#        self.loss_func = nn.BCELoss(size_average=False)
        

    def replay_train(self, policyNet, targetNet, train_batch, gamma=0.999):
        
        self.policyNet.train()
                
#        state_li = []#        
#        for state, action, reward, next_state, terminal in train_batch: 
#            state_li.append(next_state)
        
#        state_stack = np.empty(0).reshape(0, self._input_size)  # (0,4)  
#        for state, action, reward, next_state, terminal in train_batch:            
#            state_stack = np.vstack([state_stack, state])
        
        batch = self.transition(*zip(*train_batch))
        
        state_tensor = torch.tensor( batch.state, device=self.device, dtype=torch.float )
        action_tensor = torch.tensor( batch.action, device=self.device, dtype=torch.long )
        reward_tensor = torch.tensor( batch.reward, device=self.device, dtype=torch.float )
        next_state_tensor = torch.FloatTensor(batch.next_state).to(self.device)
        terminal_tensor = torch.FloatTensor(batch.terminal).to(self.device)

        # Compute max_a'Q(S_{t+1}, a', theta-) for all next states.
        maxQ_target_nextS = self.targetNet(next_state_tensor).max(1)[0].detach()

        # R_{t+1} + gamma * max_a'Q(S_{t+1}, a', theta-)
        y = reward_tensor + gamma * maxQ_target_nextS.unsqueeze(1) * terminal_tensor
        
        #Q(S_t, A_t, theta)
        Q_policy = self.policyNet(state_tensor).gather(1, action_tensor)
        
        self.policyOptimizer.zero_grad()        
        # MSE = ( y - Q(S_t, A_t, theta) )^2        
#        loss = F.smooth_l1_loss(Q_policy, y) 
        loss = self.loss_func(Q_policy, y)        
        loss.backward()
#        for param in policyNet.parameters():
#            param.grad.data.clamp_(-1, 1)
        self.policyOptimizer.step()     
        
        # Train our network using target and predicted Q values on each episode
        return loss
    
    
    def select_action(self, state):
        
        self.policyNet.eval()
#        global steps_done
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if random.random() > self.eps_threshold:
            with torch.no_grad():
                action = self.policyNet(state).max(1)[1].view(1, 1)
                return action.item()
        else:
            return self.env.action_space.sample()
#            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
    
    
    def train(self, max_episodes):
            
        # store the previous observations in replay memory
        replay_buffer = deque()        
        start_time = time.perf_counter()
        
        current_max_step = self.args.initialStepForOptimalModel
        
        # train my model
        for episode in range(max_episodes):

#            self._e = 1. / ((episode / 100.) + 1)

            terminal = False
            step_count = 0
            loss = 0.0
            state = self.env.reset()              
             
            while not terminal:
#                self.env.render()
                    
#                state_tensor = torch.from_numpy(state).to(self.device).view(-1,self._input_size)
#                state_tensor = torch.FloatTensor(state.reshape([-1, self._input_size])).to(self.device)
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float).view(-1,self._input_size)
                action = self.select_action(state_tensor)
                    
                # Get new state and reward from environment
                next_state, reward, terminal, _ = self.env.step(action)  # S_(t+1), R_t, terminal
                
                terminalNo = 1
                if terminal:    # big penalty   
                    terminalNo = 0
#                    reward = -100

                # Save the experience to our buffer (Replay Memory)
                replay_buffer.append((state, [action], [reward], next_state, [terminalNo]))  # (S, A, R, S, terminal)
                if len(replay_buffer) > self.args.replayMemory:
                    replay_buffer.popleft()
                
                state = next_state      
                step_count += 1                
                
#                if len(replay_buffer) > self.args.batch_size:
#                    minibatch = random.sample(replay_buffer, self.args.batch_size)      # Get a random batch of experiences.
#                    loss = self.replay_train(self.policyNet, self.targetNet, minibatch)
            
            self.env.close()
            
            if step_count > current_max_step : 
                current_max_step = step_count
                # save model and parameters 
                torch.save(self.policyNet, self.saveModelPath)                      # save model and parameters
                torch.save(self.policyNet.state_dict(), self.saveParamsPath ) # It saves only the model parameters (recommended)
                torch.save(self.policyNet, self.optimalModelSavePath)                      # save model and parameters
                torch.save(self.policyNet.state_dict(), self.optimalParamsSavePath ) # It saves only the model parameters (recommended)
                print('=====================================================================')
                print("Episode: {}  steps: {}  <== Good enough!!!!!!!!!!".format(episode, step_count) )
                print('Model and parameters have been saved')
                print('=====================================================================')
            
            else:
                print("Episode: {}  steps: {}".format(episode, step_count) )
            
            if episode % 10 == 0 and len(replay_buffer) >= self.args.batch_size:  
                for i in range(50):
                    minibatch = random.sample(replay_buffer, self.args.batch_size)      # Get a random batch of experiences.
                    loss = self.replay_train(self.policyNet, self.targetNet, minibatch)               
                self.targetNet.load_state_dict(self.policyNet.state_dict())
                print('=====================================================================')   
                print("Eps threshold :", self.eps_threshold)
                print("Loss :", loss)
                print('=====================================================================')
            
#            if episode > 10:
#                nCountTest = self.test(use_render=False)
#                if nCountTest > current_max_step : 
#                    current_max_step = nCountTest
#                    # save model and parameters 
#                    torch.save(self.policyNet, self.saveModelPath)                      # save model and parameters
#                    torch.save(self.policyNet.state_dict(), self.saveParamsPath ) # It saves only the model parameters (recommended)
#                    torch.save(self.policyNet, self.optimalModelSavePath)                      # save model and parameters
#                    torch.save(self.policyNet.state_dict(), self.optimalParamsSavePath ) # It saves only the model parameters (recommended)
#                    print('=====================================================================')
#                    print("Episode: {}  steps: {}  <== Good enough!!!!!!!!!!".format(episode, nCountTest) )
#                    print('Model and parameters have been saved')
#                    print('=====================================================================')
#                
#                else:
#                    print("Episode: {}  steps: {}".format(episode, nCountTest) )
                    
          
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
#            self.bot_play(self.policyNet)
            
        
    def bot_play(self, policyNet, use_render=True):
        # See our trained network in action
        policyNet.eval()
        state = self.env.reset()
        reward_sum = 0
        stepCount = 0
        
        while True:
            if use_render:
                self.env.render()

#            state_tensor = torch.FloatTensor(np.reshape(state, [-1, self._input_size])).to(self.device)                   
#            Qsa = self.policyNet(state_tensor)
#            action = torch.max(Qsa.cpu(), 1)[1].numpy()[0]

            state_tensor = torch.FloatTensor(state.reshape(-1, self._input_size)).to(self.device)                   
            with torch.no_grad():
                Qsa = policyNet(state_tensor)
            action = Qsa.max(1)[1]
            action = action.squeeze(0)
            action = action.cpu().numpy()

#            print(action)
#            print(Qsa)
#            print(torch.max(Qsa.cpu(),1))
#            print(Qsa.cpu().max())
#            print(Qsa.max(0))
            
            state, reward, terminal, _ = self.env.step(action)
            reward_sum += reward
            stepCount += 1
            if terminal:
#                print("Last Step: {}".format(stepCount), "Total score: {}".format(reward_sum))
#                if stepCount>=5000:
#                    print("Very Good!!!!!!!!!")      
##                    break                
                break
                    
        self.env.close()
        return stepCount
                
    def test(self, use_render=True):  
                  
#        game2 = wrappers.Monitor(self._game, 'gym-results', force=True)        
#        for i in range(200):            
        return self.bot_play(self.policyNet, use_render)
  
    
    

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
    
def main():
    
    print("Torch vaersion :", torch.__version__)
    
    # GPU check
    useGPU = torch.cuda.is_available()
    
    if useGPU :
        deviceNo = torch.cuda.current_device()
        print("GPU_is_available.")
        print("DeviceNo :",  deviceNo)
        print(torch.cuda.device(deviceNo))
        print("Device_count :", torch.cuda.device_count())
#        print(torch.cuda.get_device_name(0))
#        print("Device_capability :", torch.cuda.get_device_capability(deviceNo))
#        print("Device_max_memory :", torch.cuda.max_memory_allocated(deviceNo))
#        print("Device_max_memory_cached :", torch.cuda.max_memory_cached(deviceNo))
        
    else :
        print("There are no GPU.")
        
        
    args = dotdict({
            'training' : True ,
            'isGPU' : True,         # False, # True,
            'load_model': True,        
            'save_folder_file': ("Cartpole0/",'checkpoint1'),       
            'load_folder_file': ("Cartpole0/",'checkpoint1'),
            'replayMemory' : 50000,          
            'initialStepForOptimalModel' : 50000,
            'maxEpisodes': 5000,
            'batch_size' : 128,
            'learning_rate' : 1e-1,                            
            })
           
    if useGPU==False and args.isGPU==True:
        args.isGPU = False
        print("GPU is not availabe.")
    if args.isGPU==False:
        print("Runing by CPU")
        
  
    gym.envs.register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        reward_threshold=9750,
    )
    game = gym.make('CartPole-v2')
#    game = gym.make('CartPole-v0')
#    game.reset()

    cartPole = DQN2015(game, args)
    
    if(args.training==True):        	                
        cartPole.train(args.maxEpisodes)
        
    elif(args.training==False):         
        cartPole.test(use_render=True)
        
    print('Complete')
    

if __name__ == "__main__":
    main()