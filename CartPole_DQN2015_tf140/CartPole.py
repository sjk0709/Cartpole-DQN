# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import gym 
from gym.envs.registration import register


import sys, os, tty
import termios
import time

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

inkey = _Getch()

# MACROS
LEFT = 0
RIGHT = 1

# Key mapping
arrow_keys = {
        'a' : LEFT,
        'd' : RIGHT}
    

#    # MACROS
#LEFT = 0
#DOWN = 1
#RIGHT = 2
#UP = 3
#
## Key mapping
#arrow_keys = {
#        '\x1b[A': UP,
#        '\x1b[B' : DOWN,
#        '\x1b[C' : RIGHT,
#        '\x1b[D' : LEFT}
#    

    
# Register Frozenlaek with is_slippery False
#register(
#        id='FrozenLake-v3', 
#        entry_point='gym.envs.toy_text:FrozenLakeEnv',
#        kwargs={'map_name':'4x4', 'is_slippery': False}
#)
#



    
# Register Frozenlaek with is_slippery False
register(
        id='CartPole-v2', 
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
        reward_threshold=10000000.0,
)

    
env = gym.make("CartPole-v2")
env.mode='human'
#env.configure(batch_mode=False)
env.reset()



human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause

    if key==65361:
        key=48
    elif key==65363:
        key=49    
    
    a = int( key - ord('0') )
    
#    if a <= 0 or a >= ACTIONS: return
    if a < 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    
    if key==65361:
        key=48
    elif key==65363:
        key=49 
    
    a = int( key - ord('0') )
    
    if a < 0 or a >= ACTIONS: return
    human_agent_action = a
    
#    if a <= 0 or a >= ACTIONS: return        
#    if human_agent_action == a:
#        human_agent_action = 0
#totalReward = 0
#key = inkey()
#if key not in arrow_keys.keys():
#    print("Game aborted!")    
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
#            print(a)
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        
#        if r != 0:
#            print("reward %0.3f" % r)
        
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        
        if total_reward>=100: 
            print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
            return False
    
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    
    

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: 
        env.close()
        break    
#
#while True:
#    
#    
#    key = inkey()
#    if key not in arrow_keys.keys():
#        print("Game aborted!")
#        key = 'a'    
#        break
#    
#    time.sleep(0.1)
#    env.render()
#    
#    action = arrow_keys[key]       
#    action = env.action_space.sample()
#    print(action)
#    
#    state, reward, done, info = env.step(action)
##    env.render()
#    totalReward += reward
#    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)
#    
#    while human_sets_pause:
#        env.render()
#        time.sleep(0.1)
#    time.sleep(0.1)
#    
#    if done:
#        print("Finished width reward", totalReward)
#        env.close()
#        break



