#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random

import gym
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# In[2]:


import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import getopt
import random
import sys
from collections import deque

# make sure the root path is in system path
# from pathlib import Path
# base_dir = Path(__file__).resolve().parent.parent
# sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
#from torch_training.dueling_double_dqn import Agent

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

#import torch_training.Nets
from importlib_resources import path

# In[3]:


from IPython.display import clear_output
import matplotlib.pyplot as plt
from gym import spaces

use_cuda = torch.cuda.is_available()
device   = "cpu" #torch.device("cuda" if use_cuda else "cpu")
device

gym_logger.setLevel(logging.CRITICAL)

training=True
weights = True
visuals = False

if weights:
    trialstart = 600
    weightFile = f"actorcritic_checkpoint{trialstart}.pth"
else:
    trialstart = 1

cuda = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Parameters for the Environment
x_dim = 35
y_dim = 35
n_agents = 10

# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.05,  # Percentage of defective agents
                   'malfunction_rate': 100,  # Rate of malfunction occurence
                   'min_duration': 20,  # Minimal duration of malfunction
                   'max_duration': 50  # Max duration of malfunction
                   }

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We set the number of episodes we would like to train on
if 'n_trials' not in locals():
    n_trials = 15000


# In[6]:


num_features_per_node = 11 #env_Orig.obs_builder.observation_dim
tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes

#print("State Size:",state_size)
#print("Feature Size:", num_features_per_node)
# The action space of flatland is 5 discrete actions
action_size = 5


# In[54]:


#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs
#     def get_num_agents(self):
#         return self[0].get_num_agents()

class RailEnvWrapper(RailEnv):
    def __init__(self,**kwargs):
        super(RailEnvWrapper, self).__init__(**kwargs)
        
        num_features_per_node = self.obs_builder.observation_dim
        tree_depth = 2
        nr_nodes = 0
        for i in range(tree_depth + 1):
            nr_nodes += np.power(4, i)
        state_size = num_features_per_node * nr_nodes        
        self.observation_space =  spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,1), dtype=np.float32) 
        self.action_space = spaces.Discrete(action_size)

#     def action(self, action):
#         if random.random() < 0.1:
#             print("Random!")
#             return self.action_space.sample()
#         return action
    
    def reset(self):
        obs, info = super().reset(True, True)
        agent_obs = [None] * self.get_num_agents()
        #agent_next_obs = [None] * env.get_num_agents()
        #agent_obs_buffer = [None] * self.get_num_agents()

        for a in range(self.get_num_agents()):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10).astype('float32')
                #agent_obs_buffer[a] = agent_obs[a].copy()
        return agent_obs,info
    
    def step(self, action_dict):
        #print(action_dict)
        obs, rewards, dones, infos = super().step(action_dict)
        agent_obs = [None] * self.get_num_agents()
        for a in range(self.get_num_agents()):
            if obs[a]:
                #print("Obs")
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
            #else: print("No Obs")
        return agent_obs,rewards,dones,infos
    
env_Orig = RailEnvWrapper(width=x_dim,
      height=y_dim,
      rail_generator=sparse_rail_generator(max_num_cities=3,
                                           # Number of cities in map (where train stations are)
                                           seed=1,  # Random seed
                                           grid_mode=False,
                                           max_rails_between_cities=2,
                                           max_rails_in_city=3),
      schedule_generator=sparse_schedule_generator(speed_ration_map),
      number_of_agents=n_agents,
      stochastic_data=stochastic_data,  # Malfunction data generator
      obs_builder_object=TreeObservation)

env = copy.deepcopy(env_Orig)

# After training we want to render the results so we also load a renderer
env_renderer = RenderTool(env, gl="PILSVG", )

# And some variables to keep track of the progress
action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()
agent_obs_buffer = [None] * env.get_num_agents()
agent_action_buffer = [2] * env.get_num_agents()
cummulated_reward = np.zeros(env.get_num_agents())
update_values = [False] * env.get_num_agents()



#from common.multiprocessing_env import SubprocVecEnv

num_envs = 2

def make_env():
    def _thunk():
        env_Orig = RailEnvWrapper(width=x_dim,
              height=y_dim,
              rail_generator=sparse_rail_generator(max_num_cities=3,
                                                   # Number of cities in map (where train stations are)
                                                   seed=1,  # Random seed
                                                   grid_mode=False,
                                                   max_rails_between_cities=2,
                                                   max_rails_in_city=3),
              schedule_generator=sparse_schedule_generator(speed_ration_map),
              number_of_agents=n_agents,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=TreeObservation)

        env = copy.deepcopy(env_Orig)

        # After training we want to render the results so we also load a renderer
        #env_renderer = RenderTool(env, gl="PILSVG", )
        
        return env

    return _thunk

#envs = [make_env() for i in range(num_envs)] # make_env()()  #
#envs = SubprocVecEnv(envs)

envs = make_env()()

env_renderer = RenderTool(envs, gl="PILSVG", )

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.n
print(num_inputs,num_outputs)

#Hyper params:
hidden_size = 256
lr          = 3e-4

# And the max number of steps we want to take per episode
max_steps = int(4 * 2 * (20 + env.height + env.width))

columns = ['Agents', 'X_DIM', 'Y_DIM', 'TRIAL_NO', 'SCORE',
           'DONE_RATIO', 'STEPS', 'ACTION_PROB']
dfAllResults = pd.DataFrame(columns=columns)

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)

if weights:
    with open('./Nets/'+ weightFile,"rb") as file_in:
        model.load_state_dict(torch.load(file_in))

optimizer = optim.Adam(model.parameters())

for trials in range(trialstart, n_trials + 1):

    max_frames = 20000
    test_rewards = []

    newstate, newinfo = envs.reset()
    env_renderer.reset()

    # Reset score and done
    score = 0
    env_done = 0
    num_steps = 1
    state = newstate
    step = 0

    while True:

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        rewardVals = 0
        maskVals = 0
        entropy = 0
        log_prob = 0
        vals = 0
        next_vals = 0

        for _ in range(num_steps):
            # Action
            for a in range(n_agents):
                if newinfo['action_required'][a]:
                    # If an action is require, we want to store the obs a that step as well as the action
                    update_values[a] = True
                    state[a] = torch.FloatTensor(state[a]).to(device).unsqueeze(0)
                    dist, value = model(state[a])
                    action = dist.sample().squeeze()

                    log_prob += dist.log_prob(action).squeeze().unsqueeze(0)
                    entropy += dist.entropy().mean()
                    vals += value.squeeze().unsqueeze(0)

                    actionVal = action.cpu().numpy()
                    action_prob[actionVal] += 1
                else:
                    #print("No Action")
                    update_values[a] = False
                    action = 0

                action_dict.update({a: actionVal})
            if (sum(newinfo['action_required'].values()) == 0):
                #print("check")
                break

    #         action_dict_list = []
    #         for i in range(num_envs):
    #             for a in range(n_agents):
    #                 action_dict = dict()
    #                 action_dict.update({a:actionVal[a]})
    #                 action_dict_list.append(action_dict)

    #         next_state, reward, done, _ = envs.step(np.array(action_dict_list))

            #print(action_dict)
            next_state, reward, done, newinfo = envs.step(action_dict)

            if visuals:
                env_renderer.render_env(show=True, show_predictions=True, show_observations=False)
            step+=1
            # # Copy observation
            # if done['__all__']:
            #     print("done")
            #     env_done = 1
            #     break

            for a in range(n_agents):

                #rewardVals += torch.FloatTensor(np.array(reward[a])[np.newaxis,...]).to(device)
                #maskVals += torch.FloatTensor(np.array(1 - done[a])[np.newaxis,...]).unsqueeze(1).to(device)

                rewardVals += torch.FloatTensor(np.array(float(reward[a]))).to(device)

                score += reward[a] / n_agents

                maskVals += torch.FloatTensor(np.array(float(1 - done[a]))).to(device)
                if newinfo['action_required'][a]:
                    state[a] = next_state[a]

            entropy = entropy / n_agents #.mean().unsqueeze(0)# / n_agents
            log_probs.append(log_prob / n_agents)  #log_prob.mean().unsqueeze(0)
            values.append(vals / n_agents) #vals.mean().unsqueeze(0)

            rewardVals = rewardVals / n_agents
            maskVals = maskVals / n_agents
            rewards.append(rewardVals)
            masks.append(maskVals)

        if (sum(newinfo['action_required'].values()) != 0):

            for a in range(n_agents):
                if newinfo['action_required'][a]:
                    next_state[a] = torch.FloatTensor(next_state[a]).to(device).unsqueeze(0)
                    _, next_values = model(next_state[a])
                    next_vals += next_values


            #     # Copy observation
            # if done['__all__']:
            #     env_done = 1
            #     break

            next_value = next_vals / n_agents

            returns = compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            next_state, reward, done, newinfo = envs.step(action_dict)
            if visuals:
                env_renderer.render_env(show=True, show_predictions=True, show_observations=False)
            # Copy observation
        if done['__all__'] or step >= max_steps:
            env_done = 1
            break

    # Collection information about training
    tasks_finished = 0
    for current_agent in envs.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1
    done_window.append(tasks_finished / max(1, env.get_num_agents()))
    scores_window.append(score / max_steps)  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    print(
        '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tsteps: {:.2f} \t Action Probabilities: \t {}'.format(
            n_agents, x_dim, y_dim,
            trials,
            np.mean(scores_window),
            100 * np.mean(done_window),
            step, action_prob / np.sum(action_prob)), end=" ")

    data = [[n_agents, x_dim, y_dim,
            trials,
            np.mean(scores_window),
            100 * np.mean(done_window),
            step, action_prob / np.sum(action_prob)]]


    dfCur = pd.DataFrame(data,columns=columns)
    dfAllResults = pd.concat([dfAllResults,dfCur])

    dfAllResults.to_csv(f'ACTORCRITIC_TrainingResults_{n_agents}_{x_dim}_{y_dim}.csv',index=False)

    if trials % 100 == 0:
        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tsteps: {:.2f} \t Action Probabilities: \t {}'.format(
                n_agents, x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                step, action_prob / np.sum(action_prob)))
        torch.save(model.state_dict(),
                   './Nets/actorcritic_checkpoint' + str(trials) + '.pth')




