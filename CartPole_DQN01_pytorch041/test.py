# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register
#from gym import wrappers

import sys, os, time
sys.path.append(os.pardir)  # parent directory
import numpy as np

import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
#import torch.nn.functional as F
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
from torch.autograd import Variable

#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
#import matplotlib.pyplot as plt


def numpyToTorchVariable( array, isGPU=True):    

    if isGPU:
        return Variable(torch.FloatTensor(array)).cuda()
    else :
        return Variable(torch.FloatTensor(array)).cpu()        
   
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
        

    register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
        reward_threshold=975,
    )
    env = gym.make('CartPole-v2')
#    env = gym.make('CartPole-v0')

    args = dotdict({
            'isGPU' : False,         # False, # True,       
            'load_folder_file': ("Cartpole0/",'optimal'),     
            'nMaxStep': 10000,                       
            })
           
    if useGPU==False and args.isGPU==True:
        args.isGPU = False
        print("GPU is not availabe.")
    if args.isGPU==False:
        print("Runing by CPU")
      
        
    # Constants defining our neural network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print('input_size : ', input_size)
    print('output_size : ', output_size)
    
    state = env.reset()
    reward_sum = 0        
    stepCount = 0
     
    
#    modelPath = 'models/' + args.load_folder_file[0] + args.load_folder_file[1] + '_all.pkl'
    modelPath = 'models/' + args.load_folder_file[0] + args.load_folder_file[1] + '_all.pkl'
    try:       
        policyNet = torch.load(modelPath, map_location=lambda storage, location: storage)                                       
 
        print("\n--------" + modelPath + " is restored.--------\n")
        
    except:
        print("\n--------There are no models.--------\n")    
    policyNet.cuda() if args.isGPU else policyNet.cpu()
    policyNet.eval()
#    from network import NetworkJK
#    policyParamsPath = 'models/' + args.load_folder_file[0] + args.load_folder_file[1] + '_policyParams.pkl'
#    policyNet.load_state_dict(torch.load(policyParamsPath))      # it loads only the model parameters (recommended)                  
    
    while True:
        env.render()     
        
        state_torch = numpyToTorchVariable( np.reshape(state, [-1, input_size]) , args.isGPU)                   
        Qsa = policyNet(state_torch)           
            
        action = torch.max(Qsa.cpu(), 1)[1].data.numpy()[0]
        
        state, reward, terminal, _ = env.step(action)
        
        reward_sum += reward
        stepCount += 1
        
        if terminal:
            print("Last Step: {}".format(stepCount), "Total score: {}".format(reward_sum))
            if stepCount>=args.nMaxStep:
                print("Very Good!!!!!!!!!")       
            break
    env.close()
    


if __name__ == "__main__":
    main()